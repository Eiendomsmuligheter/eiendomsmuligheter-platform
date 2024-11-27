require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { analyzeProperty } = require('./services/openaiService');
const connectDB = require('./config/database');
const User = require('./models/User');
const Analysis = require('./models/Analysis');

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Authentication Middleware
const auth = async (req, res, next) => {
    try {
        const token = req.header('Authorization').replace('Bearer ', '');
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        const user = await User.findOne({ _id: decoded.userId });
        
        if (!user) {
            throw new Error();
        }
        
        req.user = user;
        next();
    } catch (error) {
        res.status(401).json({ error: 'Vennligst autentiser deg' });
    }
};

// Routes
app.post('/api/register', async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = new User({ email, password });
        await user.save();
        
        const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET);
        res.json({ user, token });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

app.post('/api/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await User.findOne({ email });
        
        if (!user || !(await user.comparePassword(password))) {
            throw new Error('Ugyldig innlogging');
        }
        
        const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET);
        res.json({ user, token });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

app.post('/api/analyze', auth, async (req, res) => {
    try {
        const { address } = req.body;
        
        // Verify user has available credits
        if (req.user.subscription === 'free' && req.user.analysisCredits < 1) {
            throw new Error('Ingen gjenværende kreditter');
        }
        
        // Create analysis record
        const analysis = new Analysis({
            user: req.user._id,
            address,
            status: 'pending'
        });
        await analysis.save();
        
        // Fetch property data (implement API call to eiendomsregister)
        const propertyData = await fetchPropertyData(address);
        
        // Perform AI analysis
        const aiAnalysis = await analyzeProperty({
            address,
            ...propertyData
        });
        
        // Update analysis with results
        analysis.propertyData = propertyData;
        analysis.aiAnalysis = aiAnalysis;
        analysis.status = 'completed';
        await analysis.save();
        
        // Deduct credit if applicable
        if (req.user.subscription === 'free') {
            req.user.analysisCredits -= 1;
            await req.user.save();
        }
        
        res.json({
            success: true,
            message: "Analyse fullført",
            data: analysis
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: "En feil oppstod under analysen",
            error: error.message
        });
    }
});

app.post('/api/subscribe', auth, async (req, res) => {
    try {
        const { paymentMethodId, plan } = req.body;
        
        // Create or update Stripe customer
        let customer;
        if (req.user.stripeCustomerId) {
            customer = await stripe.customers.retrieve(req.user.stripeCustomerId);
        } else {
            customer = await stripe.customers.create({
                email: req.user.email,
                payment_method: paymentMethodId,
                invoice_settings: {
                    default_payment_method: paymentMethodId
                }
            });
            req.user.stripeCustomerId = customer.id;
            await req.user.save();
        }
        
        // Create subscription
        const subscription = await stripe.subscriptions.create({
            customer: customer.id,
            items: [{ price: process.env[`STRIPE_${plan.toUpperCase()}_PRICE_ID`] }],
            expand: ['latest_invoice.payment_intent']
        });
        
        res.json({
            subscriptionId: subscription.id,
            clientSecret: subscription.latest_invoice.payment_intent.client_secret
        });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

// User dashboard data
app.get('/api/dashboard', auth, async (req, res) => {
    try {
        const analyses = await Analysis.find({ user: req.user._id })
            .sort({ createdAt: -1 })
            .limit(10);
            
        res.json({
            user: req.user,
            analyses,
            subscription: {
                plan: req.user.subscription,
                credits: req.user.analysisCredits
            }
        });
    } catch (error) {
        res.status(400).json({ error: error.message });
    }
});

// Healthcheck endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK', timestamp: new Date() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server kjører på port ${PORT}`);
});