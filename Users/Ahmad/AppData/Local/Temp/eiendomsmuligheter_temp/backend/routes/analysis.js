const express = require('express');
const router = express.Router();
const Analysis = require('../models/Analysis');
const auth = require('../middleware/auth');
const { analyzeProperty } = require('../services/openaiService');

// Get all analyses for a user
router.get('/', auth, async (req, res) => {
    try {
        const analyses = await Analysis.find({ user: req.user.id });
        res.json(analyses);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Serverfeil' });
    }
});

// Create new analysis
router.post('/', auth, async (req, res) => {
    try {
        const { address, propertyData } = req.body;

        // Check if user has enough credits
        const user = req.user;
        if (user.analysisCredits <= 0 && user.subscription === 'free') {
            return res.status(403).json({ message: 'Ikke nok kreditter' });
        }

        // Create initial analysis
        let analysis = new Analysis({
            user: req.user.id,
            address,
            propertyData,
            status: 'pending'
        });

        await analysis.save();

        // Perform AI analysis
        const aiResult = await analyzeProperty(propertyData);

        // Update analysis with AI results
        analysis.aiAnalysis = aiResult;
        analysis.status = 'completed';
        analysis.possibilities = parsePossibilities(aiResult.recommendation);

        await analysis.save();

        // Deduct credit if user is on free plan
        if (user.subscription === 'free') {
            user.analysisCredits -= 1;
            await user.save();
        }

        res.json(analysis);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Serverfeil ved analyse' });
    }
});

// Get specific analysis
router.get('/:id', auth, async (req, res) => {
    try {
        const analysis = await Analysis.findOne({
            _id: req.params.id,
            user: req.user.id
        });

        if (!analysis) {
            return res.status(404).json({ message: 'Analyse ikke funnet' });
        }

        res.json(analysis);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Serverfeil' });
    }
});

// Helper function to parse AI recommendations into structured possibilities
function parsePossibilities(recommendation) {
    // This is a simplified implementation - you would need to enhance this
    // based on the actual structure of your AI responses
    const possibilities = [];
    
    // Example: Extract basement apartment possibility
    if (recommendation.toLowerCase().includes('kjellerleilighet')) {
        possibilities.push({
            type: 'kjellerleilighet',
            feasibility: 'mulig',
            estimatedCost: '500000-800000',
            requirements: ['byggetillatelse', 'teknisk godkjenning'],
            description: 'Mulighet for kjellerleilighet med egen inngang'
        });
    }

    // Add more parsing logic for other types of possibilities

    return possibilities;
}

module.exports = router;