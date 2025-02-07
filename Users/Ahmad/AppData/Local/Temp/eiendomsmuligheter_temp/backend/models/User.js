const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
    email: {
        type: String,
        required: true,
        unique: true
    },
    password: {
        type: String,
        required: true
    },
    subscription: {
        type: String,
        enum: ['free', 'standard', 'pro'],
        default: 'free'
    },
    stripeCustomerId: {
        type: String
    },
    analysisCredits: {
        type: Number,
        default: 0
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('User', userSchema);