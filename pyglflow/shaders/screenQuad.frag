#version 430
    in vec3 newColor;
    in vec2 outTexCoords;
    out vec4 outColor;
	
    layout (binding = 0) uniform sampler2D samplerTex;
    layout (binding = 1) uniform sampler2D samplerProcessedTex;

	uniform int sliderR;
	uniform int sliderG;
	uniform int sliderB;

    uniform int renderType;

    void main()
    {

        vec4 col = vec4(0,0,0,1);

        if (renderType == 0) {
            // newcolor just shows you how to pass values through the shader stages
            //outColor = vec4(newColor, 1.0f);
            col.xyz = textureLod(samplerTex, outTexCoords, float(sliderR)).xyz  * 1.0f;

        }
        else if (renderType == 1) {
            col.xy = textureLod(samplerProcessedTex, outTexCoords, float(sliderR)).xy * 10.0f;

        }
        else if (renderType == 2) {
            
            col.xy = textureLod(samplerProcessedTex, outTexCoords, float(sliderR)).xy * 10.0f;
            float mag = sqrt(col.x * col.x + col.y * col.y);
            float ang = atan(col.x, col.y);

            ang -= 1.57079632679f;

            if (ang < 0.0f) {
                ang += 6.28318530718f;
            }

            ang /= 6.28318530718f;

            ang = 1.0f - ang;

            vec4 K = vec4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
            vec3 p = abs(fract(ang + K.xyz) * 6.0 - K.www);

            vec3 rgb = mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), mag * ((float(sliderR) + 1.0f) / 1.0f));

            col = vec4((1.0f - rgb) * 1.0f, mag > 0.5f ? (mag < 0.5 ? mag / 0.5 : 1.0) : 0.0 * 0.5);

        }

		outColor = col;
    }