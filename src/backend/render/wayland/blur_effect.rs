use std::{
    borrow::{Borrow, BorrowMut},
    sync::{LazyLock, Mutex},
};

use glam::{Affine2, Mat3, Vec2};
use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            Bind, BlitFrame, Color32F, ContextId, Frame, FrameContext, ImportAll, Offscreen,
            Renderer, Texture, TextureFilter,
            element::{Element, Id, Kind, RenderElement},
            gles::{
                GlesError, GlesFrame, GlesRenderer, GlesTexProgram, GlesTexture, Uniform,
                UniformName, UniformType, UniformValue, ffi,
            },
            sync::SyncPoint,
            utils::{CommitCounter, DamageSet},
        },
    },
    utils::{
        Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform, user_data::UserDataMap,
    },
    wayland::compositor::SurfaceData,
};
use tracing::trace;

use crate::{
    backend::render::{element::AsGlowRenderer, wayland::clipped_surface::ClippingShader},
    wayland::handlers::background_effect::ComputedBlurRegionCachedState,
};

pub static BLUR_DOWNSAMPLE_SHADER: &str = include_str!("../shaders/blur_downsample.frag");
pub static BLUR_UPSAMPLE_SHADER: &str = include_str!("../shaders/blur_upsample.frag");

const NOISE: f32 = 0.03;
const MAX_STEPS: usize = 15;
const PARTIAL_FRACTION: f64 = 0.85;

#[derive(Debug, Clone, Copy, PartialEq)]
struct BlurParameters {
    passes: usize,
    offset: f64,
    extended_radius: i32,
}

static BLUR_PARAMS: LazyLock<Vec<BlurParameters>> = LazyLock::new(|| {
    let mut params = Vec::new();

    let mut remaining_steps = MAX_STEPS as isize;
    let offsets = [
        // min offset, max offset, extended radius to avoid artifacts
        (1.0, 2.0, 10),
        (2.0, 3.0, 20),
        (2.0, 5.0, 50),
        (3.0, 8.0, 150),
    ];

    let sum = offsets.iter().map(|(min, max, _)| *max - *min).sum::<f64>();
    for (i, (min, max, extended_radius)) in offsets.into_iter().enumerate() {
        let mut iter_num = f64::ceil((max - min) / sum * (MAX_STEPS as f64)) as usize;
        remaining_steps -= iter_num as isize;

        if remaining_steps < 0 {
            iter_num = iter_num.saturating_add_signed(remaining_steps);
        }

        let diff = max - min;
        for j in 1..=iter_num {
            params.push(BlurParameters {
                passes: i + 1,
                offset: min + (diff / iter_num as f64) * j as f64,
                extended_radius,
            });
        }
    }

    trace!("Computed blur values: {:#?}", &params);
    params
});

fn blur_textures_reusable<R>(
    context: &ContextId<GlesTexture>,
    cache: &UserDataMap,
    tex_size: Size<i32, Buffer>,
) -> bool
where
    R: AsGlowRenderer,
    R::TextureId: Send + 'static,
{
    let entry_matches = |entry: &Option<R::TextureId>| {
        entry
            .as_ref()
            .is_some_and(|tex| tex.size() == tex_size && R::tex_to_gl(context, tex).is_some())
    };

    cache
        .get::<BlurTexture<R::TextureId>>()
        .is_some_and(|texture| entry_matches(&texture.lock().unwrap()))
        && cache
            .get::<BlurOffTexture<R::TextureId>>()
            .is_some_and(|texture| entry_matches(&texture.0.lock().unwrap()))
}

#[derive(Debug, Clone)]
pub struct BlurShaders {
    down: GlesTexProgram,
    up: GlesTexProgram,
}

impl BlurShaders {
    pub fn compile(renderer: &mut GlesRenderer) -> Result<BlurShaders, GlesError> {
        let up = renderer.compile_custom_texture_shader(
            BLUR_UPSAMPLE_SHADER,
            &[
                UniformName::new("half_pixel", UniformType::_2f),
                UniformName::new("offset", UniformType::_1f),
            ],
        )?;
        let down = renderer.compile_custom_texture_shader(
            BLUR_DOWNSAMPLE_SHADER,
            &[
                UniformName::new("half_pixel", UniformType::_2f),
                UniformName::new("offset", UniformType::_1f),
            ],
        )?;

        Ok(BlurShaders { up, down })
    }

    pub fn get<R: AsGlowRenderer>(renderer: &R) -> Self {
        Borrow::<GlesRenderer>::borrow(renderer.glow_renderer())
            .egl_context()
            .user_data()
            .get::<BlurShaders>()
            .expect("Custom Shaders not initialized")
            .clone()
    }
}

type BlurTexture<T> = Mutex<Option<T>>;

struct BlurOffTexture<T>(Mutex<Option<T>>);

impl<T> Default for BlurOffTexture<T> {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

#[derive(Debug)]
pub struct BlurState {
    pub id: Id,
    pub renderer_id: Option<ContextId<GlesTexture>>,
    pub src: Size<f64, Buffer>,
    pub offset: f64,
    pub passes: usize,
    pub region: Vec<Rectangle<i32, Logical>>,
    pub commit: CommitCounter,
}

unsafe impl Send for BlurState {}
unsafe impl Sync for BlurState {}

impl Default for BlurState {
    fn default() -> Self {
        BlurState {
            id: Id::new(),
            renderer_id: None,
            src: Size::new(0., 0.),
            offset: 0.,
            passes: 0,
            region: Vec::new(),
            commit: CommitCounter::default(),
        }
    }
}

pub struct BlurElement {
    id: Id,
    commit: CommitCounter,
    src: Size<f64, Buffer>,
    extended_offset: Point<f64, Logical>,
    geometry: Rectangle<f64, Logical>,
    scaling_shaders: BlurShaders,
    render_shader: GlesTexProgram,
    region: Vec<Rectangle<i32, Logical>>,
    offset: f64,
    passes: usize,
    uniforms: Vec<Uniform<'static>>,
}

impl BlurElement {
    fn backdrop_region(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        Rectangle::new(
            self.extended_offset.to_physical_precise_round(scale),
            self.geometry.size.to_physical_precise_round(scale)
                - self
                    .extended_offset
                    .to_size()
                    .upscale(2.)
                    .to_physical_precise_round(scale),
        )
    }

    fn capture_backdrop<R>(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), R::Error>
    where
        R: AsGlowRenderer,
        R::TextureId: Send + 'static,
    {
        let transform = frame.transformation();
        let tex_size = self.src.to_i32_round();
        let glow_frame = <R as AsGlowRenderer>::glow_frame_mut(frame);
        let gles_frame = BorrowMut::<GlesFrame<'_, '_>>::borrow_mut(glow_frame);
        let mut renderer = gles_frame.renderer();

        let texture_ref = cache.get_or_insert_threadsafe(BlurTexture::<R::TextureId>::default);
        let mut texture_entry = texture_ref.lock().unwrap();
        if texture_entry.as_ref().is_some_and(|tex| {
            tex.size() != tex_size
                || R::tex_to_gl(
                    &renderer.as_ref().context_id(),
                    texture_entry.as_ref().unwrap(),
                )
                .is_none()
        }) {
            texture_entry.take();
        }
        if texture_entry.is_none() {
            let gl_texture = renderer
                .as_mut()
                .create_buffer(Fourcc::Abgr8888, tex_size)
                .map_err(R::from_gles_error)?;
            *texture_entry = Some(R::tex_from_gl(&renderer.as_ref().context_id(), gl_texture));
        }

        let mut texture = R::tex_to_gl(
            &renderer.as_ref().context_id(),
            texture_entry.as_ref().unwrap(),
        )
        .unwrap();

        let off_texture_ref =
            cache.get_or_insert_threadsafe(BlurOffTexture::<R::TextureId>::default);
        let mut off_texture_entry = off_texture_ref.0.lock().unwrap();
        if off_texture_entry
            .as_ref()
            .is_some_and(|tex: &R::TextureId| {
                tex.size() != tex_size
                    || R::tex_to_gl(&renderer.as_ref().context_id(), tex).is_none()
            })
        {
            off_texture_entry.take();
        }
        if off_texture_entry.is_none() {
            let gl_texture = renderer
                .as_mut()
                .create_buffer(Fourcc::Abgr8888, tex_size)
                .map_err(R::from_gles_error)?;
            *off_texture_entry = Some(R::tex_from_gl(&renderer.as_ref().context_id(), gl_texture));
        }
        let mut off_texture = R::tex_to_gl(
            &renderer.as_ref().context_id(),
            off_texture_entry.as_ref().unwrap(),
        )
        .unwrap();
        std::mem::drop(renderer);

        let tex_size_phys = tex_size.to_logical(1, Transform::Normal).to_physical(1);
        let _ = blit_from_active_fb(
            gles_frame,
            src,
            dst,
            transform,
            Rectangle::from_size(tex_size_phys),
            &mut texture,
        )
        .map_err(R::from_gles_error)?;

        let mut textures = [&mut texture, &mut off_texture];
        render_blur(
            gles_frame.renderer().as_mut(),
            &self.scaling_shaders,
            &mut textures,
            self.offset,
            self.passes,
            None,
        )
        .map_err(R::from_gles_error)?;

        Ok(())
    }

    fn partial_capture_window(
        &self,
        damage: &[Rectangle<i32, Physical>],
        dst: Rectangle<i32, Physical>,
        tex_size: Size<i32, Buffer>,
    ) -> Option<Rectangle<i32, Buffer>> {
        let margin = (self.passes as f64 * self.offset * 1.5).ceil() as i32 + 4;
        damage_capture_window(damage, dst, tex_size, margin)
    }

    fn capture_backdrop_partial<R>(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        window: Rectangle<i32, Buffer>,
        cache: &UserDataMap,
    ) -> Result<(), R::Error>
    where
        R: AsGlowRenderer,
        R::TextureId: Send + 'static,
    {
        let tex_size = self.src.to_i32_round();
        let transform = frame.transformation();
        let reusable = transform == Transform::Normal && {
            let glow_frame = <R as AsGlowRenderer>::glow_frame_mut(frame);
            let gles_frame = BorrowMut::<GlesFrame<'_, '_>>::borrow_mut(glow_frame);
            let context = gles_frame.renderer().as_ref().context_id();
            blur_textures_reusable::<R>(&context, cache, tex_size)
        };
        if !reusable {
            return self.capture_backdrop::<R>(frame, src, dst, cache);
        }

        let window_phys = Rectangle::new(
            Point::from((window.loc.x, window.loc.y)),
            Size::from((window.size.w, window.size.h)),
        );
        let src_sub = Rectangle::new(
            src.loc + Point::<f64, Buffer>::new(window.loc.x as f64, window.loc.y as f64),
            window.size.to_f64(),
        );
        let dst_sub = Rectangle::new(dst.loc + window_phys.loc, window_phys.size);

        let glow_frame = <R as AsGlowRenderer>::glow_frame_mut(frame);
        let gles_frame = BorrowMut::<GlesFrame<'_, '_>>::borrow_mut(glow_frame);
        {
            let renderer = gles_frame.renderer();

            let texture_ref = cache
                .get::<BlurTexture<R::TextureId>>()
                .expect("blur texture missing despite passing the reusability check");
            let texture_entry = texture_ref.lock().unwrap();
            let mut texture = R::tex_to_gl(
                &renderer.as_ref().context_id(),
                texture_entry.as_ref().unwrap(),
            )
            .unwrap();

            let off_texture_ref = cache
                .get::<BlurOffTexture<R::TextureId>>()
                .expect("blur off-texture missing despite passing the reusability check");
            let off_texture_entry = off_texture_ref.0.lock().unwrap();
            let mut off_texture = R::tex_to_gl(
                &renderer.as_ref().context_id(),
                off_texture_entry.as_ref().unwrap(),
            )
            .unwrap();
            std::mem::drop(renderer);

            let _ = blit_from_active_fb(
                gles_frame,
                src_sub,
                dst_sub,
                Transform::Normal,
                window_phys,
                &mut texture,
            )
            .map_err(R::from_gles_error)?;

            let mut textures = [&mut texture, &mut off_texture];
            render_blur(
                gles_frame.renderer().as_mut(),
                &self.scaling_shaders,
                &mut textures,
                self.offset,
                self.passes,
                Some(window),
            )
            .map_err(R::from_gles_error)?;
        }

        Ok(())
    }
}

impl BlurElement {
    pub fn from_state<R: ImportAll + AsGlowRenderer>(
        renderer: &mut R,
        state: &mut BlurState,
        geometry: Rectangle<f64, Logical>,
        output_scale: f64,
        radii: [u8; 4],
        strength: usize,
    ) -> Result<Option<Self>, R::Error> {
        let region = vec![Rectangle::from_size(geometry.size.to_i32_round())];

        Self::internal(
            renderer,
            state,
            geometry,
            &region,
            output_scale,
            radii,
            strength,
        )
    }

    pub fn from_surface<R: ImportAll + AsGlowRenderer>(
        renderer: &mut R,
        states: &SurfaceData,
        geometry: Rectangle<f64, Logical>,
        output_scale: f64,
        radii: [u8; 4],
        strength: usize,
    ) -> Result<Option<Self>, R::Error> {
        let mut blur_region_state = states.cached_state.get::<ComputedBlurRegionCachedState>();
        let Some(region) = blur_region_state.current().blur_region.as_ref() else {
            return Ok(None);
        };

        let state = states
            .data_map
            .get_or_insert_threadsafe::<Mutex<BlurState>, _>(Default::default);

        Self::internal(
            renderer,
            &mut state.lock().unwrap(),
            geometry,
            region,
            output_scale,
            radii,
            strength,
        )
    }

    pub fn internal<R: ImportAll + AsGlowRenderer>(
        renderer: &mut R,
        state: &mut BlurState,
        geometry: Rectangle<f64, Logical>,
        region: &Vec<Rectangle<i32, Logical>>,
        output_scale: f64,
        radii: [u8; 4],
        strength: usize,
    ) -> Result<Option<Self>, R::Error> {
        if strength == 0 || geometry.size.w == 0. || geometry.size.h == 0. {
            return Ok(None);
        }

        let geo = geometry.to_physical_precise_round(output_scale);
        let mut extended_geo = geo;
        let radius = BLUR_PARAMS[(strength + 2).min(MAX_STEPS - 1)].extended_radius as f64;
        extended_geo.loc -= Point::<f64, Physical>::new(radius, radius);
        extended_geo.size += Size::<f64, Physical>::new(radius, radius).upscale(2.);

        // compute input_to_geo so that it crops the extended capture radius
        let geo_scale = {
            let Scale { x, y } = geo.size / extended_geo.size;
            Affine2::from_scale(Vec2::new(x as f32, y as f32)).inverse()
        };
        let geo_translation = {
            let offset = geo.loc - extended_geo.loc;
            Affine2::from_translation(-Vec2::new(
                (offset.x / extended_geo.size.w) as f32,
                (offset.y / extended_geo.size.h) as f32,
            ))
        };
        let input_to_geo = Mat3::from(geo_scale * geo_translation);

        let uniforms = vec![
            Uniform::new("geo_size", (geometry.size.w as f32, geometry.size.h as f32)),
            Uniform::new(
                "corner_radius",
                [
                    radii[0] as f32,
                    radii[1] as f32,
                    radii[2] as f32,
                    radii[3] as f32,
                ],
            ),
            Uniform::new(
                "input_to_geo",
                UniformValue::Matrix3x3 {
                    matrices: vec![*AsRef::<[f32; 9]>::as_ref(&input_to_geo)],
                    transpose: false,
                },
            ),
            Uniform::new("noise", UniformValue::_1f(NOISE)),
        ];

        let geometry = extended_geo.to_logical(output_scale);
        let extended_offset = Point::<f64, Physical>::new(radius, radius).to_logical(output_scale);

        let renderer_id = renderer.glow_renderer().context_id();
        let src = geometry.size.to_buffer(output_scale, Transform::Normal);
        let params = &BLUR_PARAMS[strength.min(MAX_STEPS - 1)];

        let dirty = !(state
            .renderer_id
            .as_ref()
            .is_some_and(|id| id == &renderer_id)
            && state.offset == params.offset
            && state.passes == params.passes
            && &state.region == region
            && state.src == src);

        state.renderer_id = Some(renderer_id);
        state.offset = params.offset;
        state.passes = params.passes;
        state.region = region.clone();
        state.src = src;
        if dirty {
            state.commit.increment();
        }

        Ok(Some(BlurElement {
            id: state.id.clone(),
            commit: state.commit,
            src,
            geometry,
            extended_offset,
            scaling_shaders: BlurShaders::get(renderer),
            render_shader: ClippingShader::get(renderer),
            offset: state.offset,
            passes: state.passes,
            region: region
                .iter()
                .cloned()
                .map(|mut rect| {
                    rect.loc += extended_offset.to_i32_round();
                    rect
                })
                .collect(),
            uniforms,
        }))
    }
}

impl Element for BlurElement {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.commit
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        Rectangle::from_size(self.src)
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.geometry.to_physical_precise_round(scale)
    }

    fn transform(&self) -> Transform {
        Transform::Normal
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        if self.commit.distance(commit).is_none_or(|d| d > 0) {
            DamageSet::from_slice(&[self.backdrop_region(scale)])
        } else {
            DamageSet::default()
        }
    }

    fn alpha(&self) -> f32 {
        1.0
    }

    fn kind(&self) -> Kind {
        Kind::default()
    }

    fn is_framebuffer_effect(&self) -> bool {
        true
    }
}

impl<R: Renderer + AsGlowRenderer> RenderElement<R> for BlurElement
where
    R::TextureId: Send + 'static,
{
    fn capture_framebuffer(
        &self,
        frame: &mut <R>::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        cache: &UserDataMap,
    ) -> Result<(), <R>::Error> {
        let tex_size = self.src.to_i32_round();
        if let Some(window) = self.partial_capture_window(damage, dst, tex_size) {
            return self.capture_backdrop_partial::<R>(frame, src, dst, window, cache);
        }

        self.capture_backdrop::<R>(frame, src, dst, cache)
    }

    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        let src_to_geo = self.geometry.size / self.src;
        let src_log = src
            .upscale(src_to_geo)
            .to_logical(1., Transform::Normal, &Size::default());
        let scale = dst.size.to_f64() / src_log.size;

        let damage = self
            .region
            .iter()
            .flat_map(|rect| {
                let mut rect = rect.to_f64().intersection(src_log)?;
                rect.loc -= src_log.loc;
                Some(rect.to_physical_precise_round(scale))
            })
            .flat_map(|rect| damage.iter().flat_map(move |r| r.intersection(rect)))
            .collect::<Vec<_>>();
        let cache = cache.expect("Framebuffer element without cache?");

        let Some(texture) = cache.get::<BlurTexture<R::TextureId>>() else {
            return Err(R::from_gles_error(GlesError::BlitError));
        };
        let texture_ref = texture.lock().unwrap();

        if let Some(tex) = texture_ref.as_ref() {
            BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
                .override_default_tex_program(self.render_shader.clone(), self.uniforms.clone());
            frame.render_texture_from_to(
                tex,
                src,
                dst,
                &damage,
                opaque_regions,
                Transform::Normal,
                1.0,
            )?;
            BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
                .clear_tex_program_override();
        }
        Ok(())
    }
}
fn expand_rect(rect: Rectangle<i32, Buffer>, pad: i32) -> Rectangle<i32, Buffer> {
    Rectangle {
        loc: rect.loc - Point::from((pad, pad)),
        size: rect.size + Size::from((pad * 2, pad * 2)),
    }
}

fn halve_rect(rect: Rectangle<i32, Buffer>) -> Rectangle<i32, Buffer> {
    let x1 = rect.loc.x / 2;
    let y1 = rect.loc.y / 2;
    let x2 = (rect.loc.x + rect.size.w + 1) / 2;
    let y2 = (rect.loc.y + rect.size.h + 1) / 2;
    Rectangle::new(Point::from((x1, y1)), Size::from((x2 - x1, y2 - y1)))
}

fn blur_level_windows(
    tex_size: Size<i32, Buffer>,
    window: Rectangle<i32, Buffer>,
    offset: f64,
    passes: usize,
) -> Vec<Rectangle<i32, Buffer>> {
    let mut windows = Vec::with_capacity(passes + 1);
    let mut current = window;
    windows.push(current);
    for i in 0..passes {
        let down_pad = (offset / (1u64 << i) as f64 * 0.5).ceil() as i32 + 1;
        let up_pad = (offset / (1u64 << (i + 1)) as f64).ceil() as i32 + 1;
        let level_size = tex_size.downscale(1 << i);
        let next_level_size = tex_size.downscale(1 << (i + 1));
        let expanded = expand_rect(current, down_pad)
            .intersection(Rectangle::from_size(level_size))
            .unwrap_or_default();
        current = expand_rect(halve_rect(expanded), up_pad)
            .intersection(Rectangle::from_size(next_level_size))
            .unwrap_or_default();
        windows.push(current);
    }
    windows
}

fn damage_capture_window(
    damage: &[Rectangle<i32, Physical>],
    dst: Rectangle<i32, Physical>,
    tex_size: Size<i32, Buffer>,
    margin: i32,
) -> Option<Rectangle<i32, Buffer>> {
    if damage.is_empty()
        || dst.size.w.abs_diff(tex_size.w) > 1
        || dst.size.h.abs_diff(tex_size.h) > 1
    {
        return None;
    }
    let full = Rectangle::from_size(tex_size);
    let mut window: Option<Rectangle<i32, Buffer>> = None;
    for d in damage {
        // 1:1 mapping: the physical rect is equally valid texture coordinates
        let d = Rectangle::<i32, Buffer>::new(
            Point::from((d.loc.x, d.loc.y)),
            Size::from((d.size.w, d.size.h)),
        );
        let Some(d) = d.intersection(full) else {
            continue;
        };
        window = Some(match window {
            None => d,
            Some(w) => {
                let loc = Point::from((w.loc.x.min(d.loc.x), w.loc.y.min(d.loc.y)));
                let size = Size::from((
                    (w.loc.x + w.size.w).max(d.loc.x + d.size.w) - loc.x,
                    (w.loc.y + w.size.h).max(d.loc.y + d.size.h) - loc.y,
                ));
                Rectangle::new(loc, size)
            }
        });
    }
    let window = expand_rect(window?, margin).intersection(full)?;
    if window.is_empty() {
        return None;
    }
    let fraction = (window.size.w as f64) * (window.size.h as f64)
        / ((tex_size.w as f64) * (tex_size.h as f64));
    (fraction <= PARTIAL_FRACTION).then_some(window)
}

fn blit_from_active_fb(
    frame: &mut GlesFrame<'_, '_>,
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    transform: Transform,
    region: Rectangle<i32, Physical>,
    to_texture: &mut GlesTexture,
) -> Result<SyncPoint, GlesError> {
    let tex_size = to_texture.size();
    let tex_size_phys = tex_size.to_logical(1, Transform::Normal).to_physical(1);
    let fb_size = frame.output_size();

    let mut renderer = frame.renderer();
    let mut fb = renderer.as_mut().bind(to_texture)?;

    if transform != Transform::Normal {
        // We need to copy to a temporary texture to do an actual
        // render pass with `render_texture_from_to` to do the rotation.
        // dst is in screen space, but we just want to do a 1:1 copy in
        // buffer space from dst of the current fb, so we need to undo any
        // transforms for the blit.
        let dst_phys = transform.transform_rect_in(dst, &fb_size);
        let dst_buffer = dst_phys
            .to_logical(1)
            .to_buffer(1, Transform::Normal, &Size::default());
        let mut tmp_texture = renderer
            .as_mut()
            .create_buffer(Fourcc::Abgr8888, dst_buffer.size)?;
        let mut fb_tmp = renderer.as_mut().bind(&mut tmp_texture)?;
        std::mem::drop(renderer);

        let sync = frame.blit_to(
            &mut fb_tmp,
            dst_phys,
            Rectangle::from_size(dst_phys.size),
            TextureFilter::Linear,
        )?;
        frame.wait(&sync)?;
        std::mem::drop(fb_tmp);

        // now we bind the target texture with `Transform::Normal`
        // and render the temporary texture with inverse transform
        // into src.
        let mut renderer = frame.renderer();
        let mut frame = renderer
            .as_mut()
            .render(&mut fb, tex_size_phys, Transform::Normal)?;
        frame.wait(&sync)?;
        Frame::render_texture_from_to(
            &mut frame,
            &tmp_texture,
            Rectangle::from_size(dst_buffer.size.to_f64()),
            src.to_logical(1., Transform::Normal, &Size::default())
                .to_physical(1.)
                .to_i32_round(),
            &[Rectangle::from_size(dst.size)],
            &[Rectangle::from_size(dst.size)],
            transform.invert(),
            1.0,
        )?;
        std::mem::drop(tmp_texture);
        frame.finish()
    } else {
        std::mem::drop(renderer);
        frame.blit_to(&mut fb, dst, region, TextureFilter::Linear)
    }
}

fn render_blur(
    renderer: &mut GlesRenderer,
    shaders: &BlurShaders,
    textures: &mut [&mut GlesTexture; 2],
    offset: f64,
    passes: usize,
    window: Option<Rectangle<i32, Buffer>>,
) -> Result<(), GlesError> {
    let windows = window.map(|window| blur_level_windows(textures[0].size(), window, offset, passes));
    let windows = windows.as_deref();

    for i in 0..passes {
        let tex_size = textures[0].size();
        let [src_tex, target_tex] = textures;
        let mut fb = renderer.bind(*target_tex)?;

        let adjusted_tex_size = tex_size.downscale(1 << i);
        let target_tex_size = tex_size
            .downscale(1 << (i + 1))
            .to_logical(1, Transform::Normal)
            .to_physical(1);
        let half_pixel = [
            0.5 / (adjusted_tex_size.w as f32),
            0.5 / (adjusted_tex_size.h as f32),
        ];
        let (src_rect, dst_rect) = match windows {
            Some(windows) => (
                windows[i].to_f64(),
                Rectangle::new(
                    Point::from((windows[i + 1].loc.x, windows[i + 1].loc.y)),
                    Size::from((windows[i + 1].size.w, windows[i + 1].size.h)),
                ),
            ),
            None => (
                Rectangle::from_size(adjusted_tex_size.to_f64()),
                Rectangle::from_size(target_tex_size),
            ),
        };

        let mut frame = renderer.render(
            &mut fb,
            tex_size.to_logical(1, Transform::Normal).to_physical(1),
            Transform::Normal,
        )?;
        frame.clear(Color32F::new(0., 0., 0., 0.), &[dst_rect])?;
        frame.with_context(|gl| unsafe {
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_S,
                ffi::CLAMP_TO_EDGE as i32,
            );
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_T,
                ffi::CLAMP_TO_EDGE as i32,
            );
        })?;
        frame.render_texture_from_to(
            src_tex,
            src_rect,
            dst_rect,
            &[dst_rect],
            &[dst_rect],
            Transform::Normal,
            1.0,
            Some(&shaders.down),
            &[
                Uniform::new("half_pixel", half_pixel),
                Uniform::new("offset", (offset / (1 << i) as f64) as f32),
            ],
        )?;
        frame.with_context(|gl| unsafe {
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::REPEAT as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::REPEAT as i32);
        })?;
        let _ = frame.finish()?;
        std::mem::drop(fb);

        textures.swap(0, 1);
    }

    for i in 0..passes {
        let tex_size = textures[0].size();
        let [src_tex, target_tex] = textures;
        let mut fb = renderer.bind(*target_tex)?;

        let adjusted_tex_size = tex_size.downscale(1 << (passes - i));
        let target_tex_size = tex_size
            .downscale(1 << (passes - i - 1))
            .to_logical(1, Transform::Normal)
            .to_physical(1);
        let half_pixel = [
            0.5 / (adjusted_tex_size.w as f32),
            0.5 / (adjusted_tex_size.h as f32),
        ];
        let (src_rect, dst_rect) = match windows {
            Some(windows) => (
                windows[passes - i].to_f64(),
                Rectangle::new(
                    Point::from((
                        windows[passes - i - 1].loc.x,
                        windows[passes - i - 1].loc.y,
                    )),
                    Size::from((
                        windows[passes - i - 1].size.w,
                        windows[passes - i - 1].size.h,
                    )),
                ),
            ),
            None => (
                Rectangle::from_size(adjusted_tex_size.to_f64()),
                Rectangle::from_size(target_tex_size),
            ),
        };

        let mut frame = renderer.render(
            &mut fb,
            tex_size.to_logical(1, Transform::Normal).to_physical(1),
            Transform::Normal,
        )?;
        frame.clear(Color32F::new(0., 0., 0., 0.), &[dst_rect])?;
        frame.with_context(|gl| unsafe {
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_S,
                ffi::CLAMP_TO_EDGE as i32,
            );
            gl.TexParameteri(
                ffi::TEXTURE_2D,
                ffi::TEXTURE_WRAP_T,
                ffi::CLAMP_TO_EDGE as i32,
            );
        })?;
        frame.render_texture_from_to(
            src_tex,
            src_rect,
            dst_rect,
            &[dst_rect],
            &[dst_rect],
            Transform::Normal,
            1.0,
            Some(&shaders.up),
            &[
                Uniform::new("half_pixel", half_pixel),
                Uniform::new("offset", (offset / (1 << (passes - i)) as f64) as f32),
            ],
        )?;
        frame.with_context(|gl| unsafe {
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_S, ffi::REPEAT as i32);
            gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_WRAP_T, ffi::REPEAT as i32);
        })?;
        let _ = frame.finish()?;
        std::mem::drop(fb);

        textures.swap(0, 1);
    }

    // textures always end up the right way around with `self.texture` containing our final render,
    // since we render PASSES * 2 (downscale and upscale), so the number of swaps is always even.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_level_windows_stay_within_levels() {
        let tex_size = Size::from((1920, 160));
        let window = Rectangle::new(Point::from((100, 20)), Size::from((300, 40)));
        for offset in [1.0, 4.0, 8.0] {
            let windows = blur_level_windows(tex_size, window, offset, 4);
            assert_eq!(windows.len(), 5);
            for (i, w) in windows.iter().enumerate() {
                let level = Rectangle::from_size(tex_size.downscale(1 << i));
                assert_eq!(
                    *w,
                    w.intersection(level).unwrap_or_default(),
                    "level {i} window must be clamped to its level"
                );
                assert!(!w.is_empty(), "level {i} window must not be empty");
            }
        }
    }

    #[test]
    fn blur_level_windows_clamp_at_edges() {
        let tex_size = Size::from((500, 100));
        let window = Rectangle::new(Point::from((0, 0)), Size::from((10, 10)));
        let windows = blur_level_windows(tex_size, window, 8.0, 4);
        for (i, w) in windows.iter().enumerate() {
            assert!(w.loc.x >= 0 && w.loc.y >= 0, "level {i}");
            let level = tex_size.downscale(1 << i);
            assert!(w.loc.x + w.size.w <= level.w, "level {i}");
            assert!(w.loc.y + w.size.h <= level.h, "level {i}");
        }
    }

    #[test]
    fn damage_capture_window_bounding_and_margin() {
        let tex_size = Size::from((1000, 200));
        let dst = Rectangle::new(Point::from((0, 0)), Size::from((1000, 200)));

        let damage = [Rectangle::new(Point::from((400, 80)), Size::from((100, 40)))];
        let window = damage_capture_window(&damage, dst, tex_size, 30).unwrap();
        assert!(window.loc.x <= 370 && window.loc.x + window.size.w >= 530);
        assert!(window.loc.y <= 50 && window.loc.y + window.size.h >= 150);

        // multiple rects are bounded together
        let damage = [
            Rectangle::new(Point::from((100, 10)), Size::from((20, 10))),
            Rectangle::new(Point::from((800, 150)), Size::from((20, 10))),
        ];
        let window = damage_capture_window(&damage, dst, tex_size, 10).unwrap();
        assert!(window.loc.x <= 90 && window.loc.x + window.size.w >= 830);
        assert!(window.loc.y <= 0 && window.loc.y + window.size.h >= 170);

        // damage covering (nearly) everything is not worth a partial capture
        let damage = [Rectangle::new(Point::from((0, 0)), Size::from((999, 199)))];
        assert!(damage_capture_window(&damage, dst, tex_size, 30).is_none());

        // no damage, or a geometry mismatching the texture, cannot be partial
        assert!(damage_capture_window(&[], dst, tex_size, 30).is_none());
        let mismatching = Rectangle::new(Point::from((0, 0)), Size::from((500, 100)));
        let damage = [Rectangle::new(Point::from((10, 10)), Size::from((20, 20)))];
        assert!(damage_capture_window(&damage, mismatching, tex_size, 30).is_none());
    }
}
