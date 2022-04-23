from typing import Any, Tuple
import numpy as np
from nptyping import NDArray
import moderngl as MGL


VERTEX_SHADER = """
#version 330

in vec2 point_uv;
out vec2 fragment_uv;

void main() {
    gl_Position = vec4(point_uv.x * 2.0 - 1.0, point_uv.y * 2.0 - 1.0, 0.0, 1.0);
    fragment_uv = point_uv;
}
"""

FRAGMENT_SHADER = """
#version 330

in vec2 fragment_uv;
out vec4 fragment_color;

uniform sampler2D texture_color;

void main() {
    vec3 rgb = texture(texture_color, fragment_uv).rgb;
    fragment_color = vec4(rgb, 1.0);
}

"""


class UVTrianglesRenderer:
    def __init__(self, ctx: MGL.Context, output_size: Tuple[int, int]):
        self.ctx = ctx
        self.output_size = output_size
        self.shader = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.fbo = self.ctx.framebuffer(
            self.ctx.renderbuffer(self.output_size, dtype="f4")
        )
        self.background_color = (0, 0, 0)

    def __del__(self):
        self.ctx.release()
        self.fbo.release()
        self.shader.release()

    @staticmethod
    def with_standalone_ctx(output_size: Tuple[int, int]) -> "UVTrianglesRenderer":
        return UVTrianglesRenderer(
            MGL.create_standalone_context(require=330), output_size
        )

    @staticmethod
    def with_window_ctx(output_size: Tuple[int, int]) -> "UVTrianglesRenderer":
        return UVTrianglesRenderer(MGL.create_context(require=330), output_size)

    def _init_ctx_object(self, tex_coords, tri_indices, texture):
        resources = []
        tex_coords_buffer = self.ctx.buffer(tex_coords.astype("f4").tobytes())
        resources.append(tex_coords_buffer)
        tri_indices_buffer = self.ctx.buffer(tri_indices.astype("i4").tobytes())
        resources.append(tri_indices_buffer)

        texture_height = texture.shape[0]
        texture_width = texture.shape[1]
        if len(texture.shape) == 3:
            components = texture.shape[2]
        else:
            components = 1

        texture_object = self.ctx.texture(
            (texture_width, texture_height),
            components,
            texture.astype("f4").tobytes(),
            dtype="f4",
        )

        resources.append(texture_object)

        content = (tex_coords_buffer, "2f4", "point_uv")
        self.shader["texture_color"] = 0
        texture_object.use(0)
        vao = self.ctx.vertex_array(self.shader, [content], tri_indices_buffer)
        resources.append(vao)

        return vao, resources

    def _render(self, vao):
        self.fbo.use()
        self.ctx.clear(*self.background_color)
        vao.render()
        self.ctx.finish()

    def _get_fbo_image(self):
        fbo_image = self.fbo.read(dtype="f4")
        fbo_image = np.frombuffer(fbo_image, dtype="f4").reshape(
            self.output_size[1], self.output_size[0], 3
        )
        fbo_image = np.flipud(fbo_image)
        return fbo_image

    def render(
        self,
        tex_coords: NDArray[(Any, 2), float],
        tri_indices: NDArray[(Any, 3), int],
        texture: NDArray[(Any, Any, 3), float],
        flip_y: bool = True,
    ) -> NDArray[(Any, Any, 3), float]:
        assert isinstance(tex_coords, NDArray[(Any, 2), float])
        assert isinstance(tri_indices, NDArray[(Any, 3), int])
        assert isinstance(texture, NDArray[(Any, Any, 3), float])

        if flip_y:
            texture = np.flipud(texture)

        resources = []

        try:
            vao, resources = self._init_ctx_object(tex_coords, tri_indices, texture)
            self._render(vao)
            result = self._get_fbo_image()

            return result

        finally:
            for resource in resources:
                resource.release()
