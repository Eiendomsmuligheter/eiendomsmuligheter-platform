const express = require('express');
const router = express.Router();
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const auth = require('../middleware/auth');
const User = require('../models/User');

// Create subscription
router.post('/create-subscription', auth, async (req, res) => {
    try {
        const { paymentMethodId, priceId } = req.body;
        const user = await User.findById(req.user.id);

        // Create or get Stripe customer
        let customerId = user.stripeCustomerId;
        if (!customerId) {
            const customer = await stripe.customers.create({
                email: user.email,
                payment_method: paymentMethodId,
                invoice_settings: { default_payment_method: paymentMethodId }
            });
            customerId = customer.id;
            user.stripeCustomerId = customerId;
            await user.save();
        }

        // Create subscription
        const subscription = await stripe.subscriptions.create({
            customer: customerId,
            items: [{ price: priceId }],
            expand: ['latest_invoice.payment_intent']
        });

        // Update user subscription status
        user.subscription = priceId === process.env.STRIPE_PRO_PRICE_ID ? 'pro' : 'standard';
        await user.save();

        res.json({
            subscriptionId: subscription.id,
            clientSecret: subscription.latest_invoice.payment_intent.client_secret
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Feil ved oppretting av abonnement' });
    }
});

// Cancel subscription
router.post('/cancel-subscription', auth, async (req, res) => {
    try {
        const user = await User.findById(req.user.id);
        if (!user.stripeCustomerId) {
            return res.status(400).json({ message: 'Ingen aktivt abonnement' });
        }

        const subscriptions = await stripe.subscriptions.list({
            customer: user.stripeCustomerId,
            status: 'active'
        });

        if (subscriptions.data.length === 0) {
            return res.status(400).json({ message: 'Ingen aktivt abonnement' });
        }

        // Cancel subscription at period end
        await stripe.subscriptions.update(subscriptions.data[0].id, {
            cancel_at_period_end: true
        });

        res.json({ message: 'Abonnement vil bli avsluttet ved periodens slutt' });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Feil ved kansellering av abonnement' });
    }
});

// Purchase analysis credits
router.post('/purchase-credits', auth, async (req, res) => {
    try {
        const { amount, paymentMethodId } = req.body;
        const user = await User.findById(req.user.id);

        const paymentIntent = await stripe.paymentIntents.create({
            amount: amount * 100, // Convert to cents
            currency: 'nok',
            customer: user.stripeCustomerId,
            payment_method: paymentMethodId,
            confirm: true
        });

        if (paymentIntent.status === 'succeeded') {
            user.analysisCredits += amount;
            await user.save();
            res.json({ message: 'Kreditter kjøpt', credits: user.analysisCredits });
        } else {
            res.status(400).json({ message: 'Betalingsfeil' });
        }
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Feil ved kjøp av kreditter' });
    }
});

module.exports = router;