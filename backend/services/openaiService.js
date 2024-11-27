const { Configuration, OpenAIApi } = require('openai');

const configuration = new Configuration({
    apiKey: process.env.OPENAI_API_KEY
});
const openai = new OpenAIApi(configuration);

const analyzeProperty = async (propertyData) => {
    try {
        const prompt = `
        Analyser følgende eiendomsdata og gi anbefalinger for utleiemuligheter:
        
        Adresse: ${propertyData.address}
        Størrelse: ${propertyData.size} kvm
        Byggeår: ${propertyData.yearBuilt}
        Type: ${propertyData.propertyType}
        Regulering: ${propertyData.zoning}

        Vurder følgende aspekter:
        1. Mulighet for kjellerleilighet
        2. Potensial for påbygg
        3. Andre utleiemuligheter
        4. Estimerte kostnader
        5. Tekniske krav og reguleringsbestemmelser
        `;

        const completion = await openai.createCompletion({
            model: "gpt-4",
            prompt: prompt,
            max_tokens: 1000,
            temperature: 0.7
        });

        const analysis = completion.data.choices[0].text;
        
        // Parse and structure the AI response
        return {
            recommendation: analysis,
            confidenceScore: 0.85, // This would be calculated based on AI certainty
            potentialValue: calculatePotentialValue(analysis)
        };
    } catch (error) {
        console.error('OpenAI API Error:', error);
        throw new Error('Kunne ikke fullføre AI-analyse');
    }
};

const calculatePotentialValue = (analysis) => {
    // Implement logic to extract and calculate potential value from analysis
    // This is a placeholder implementation
    return 500000;
};

module.exports = {
    analyzeProperty
};