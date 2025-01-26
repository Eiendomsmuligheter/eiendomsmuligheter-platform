const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    address: {
        type: String,
        required: true
    },
    propertyData: {
        size: Number,
        yearBuilt: Number,
        propertyType: String,
        zoning: String
    },
    possibilities: [{
        type: {
            type: String,
            required: true
        },
        feasibility: String,
        estimatedCost: String,
        requirements: [String],
        description: String
    }],
    aiAnalysis: {
        recommendation: String,
        confidenceScore: Number,
        potentialValue: Number
    },
    status: {
        type: String,
        enum: ['pending', 'completed', 'failed'],
        default: 'pending'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Analysis', analysisSchema);