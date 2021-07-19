#version 430
    in vec3 newColor;
    in vec2 outTexCoords;
    out vec4 outColor;
	
    layout (binding = 0) uniform sampler2D samplerTex;
    layout (binding = 1) uniform sampler2D samplerProcessedTex;
    layout (binding = 2) uniform sampler2D gradientTex;
    layout (binding = 3) uniform usampler2D poseTex;
    layout (binding = 4) uniform sampler2D depth;

	uniform int sliderR;
	uniform int sliderG;
	uniform int sliderB;

    uniform int renderType;
    uniform vec2 depthRange;

    vec3 inferno(float t) {

    const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

vec3 plasma(float t) {

    const vec3 c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    const vec3 c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    const vec3 c2 = vec3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
    const vec3 c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    const vec3 c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

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
            float normDepth = float(textureLod(depth, vec2(outTexCoords.x, 1.0 - outTexCoords.y), float(sliderR)).x);

            float scaledDepth = smoothstep(depthRange.x, depthRange.y, normDepth);
            col.xyz = plasma(scaledDepth);

            vec4 tcol;
            tcol.xy = textureLod(samplerProcessedTex, outTexCoords, float(sliderR)).xy * 1.0f;
            float mag = sqrt(tcol.x * tcol.x + tcol.y * tcol.y);
            float ang = atan(tcol.x, tcol.y);

            ang -= 1.57079632679f;

            if (ang < 0.0f) {
                ang += 6.28318530718f;
            }

            ang /= 6.28318530718f;

            ang = 1.0f - ang;

            vec4 K = vec4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
            vec3 p = abs(fract(ang + K.xyz) * 6.0 - K.www);

            vec3 rgb = mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), mag * ((float(sliderR) + 1.0f) / 1.0f));

            tcol = vec4((1.0f - rgb) * 0.3f, mag > 0.5f ? (mag < 0.5 ? mag / 0.5 : 1.0) : 0.0 * 0.5);

            if (mag > 0.5) {
                col = tcol;
            }

        }
        else if (renderType == 3) {
            vec2 grad = textureLod(gradientTex, outTexCoords, float(3)).xy * 50.0f;
            col.xyz = vec3(grad.x * grad.y) * vec3(0.2, 0.3, 1.0);
            vec3 posePixel = textureLod(poseTex, outTexCoords, 0.0f).xyz;
            if (posePixel.x > 0 || posePixel.y > 0 || posePixel.z > 0) {
                col.xyz = posePixel;
            }
        }

		outColor = col;
    }