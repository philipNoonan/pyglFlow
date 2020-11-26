#version 430
    in vec3 newColor;
    in vec2 outTexCoords;
    out vec4 outColor;
	
    layout (binding = 0) uniform sampler2DArray samplerTex;
    layout (binding = 1) uniform sampler2D smaplerProcessedTex;

	uniform int sliderR;
	uniform int sliderG;
	uniform int sliderB;

    uniform int renderType;

    void main()
    {
        float texDataR;
		float texDataG;
        float texDataB;

        if (renderType == 0) {
            // newcolor just shows you how to pass values through the shader stages
            //outColor = vec4(newColor, 1.0f);
            texDataR = texture(samplerTex, vec3(outTexCoords, float(sliderR))).x  * 1.0f;
            texDataG = texture(samplerTex, vec3(outTexCoords, float(sliderG))).x  * 1.0f;
            texDataB = texture(samplerTex, vec3(outTexCoords, float(sliderB))).x  * 1.0f;
        }
        else if (renderType == 1) {
            vec2 tData = texture(smaplerProcessedTex, outTexCoords).xy * 1.0f;
            texDataR = tData.x;
            texDataG = tData.y;
        }

		outColor = vec4(texDataR, texDataG, texDataB, 1.0f);
    }