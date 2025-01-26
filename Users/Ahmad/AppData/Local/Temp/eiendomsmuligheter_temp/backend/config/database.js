const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('MongoDB tilkoblet');
    } catch (error) {
        console.error('MongoDB tilkoblingsfeil:', error.message);
        process.exit(1);
    }
};

module.exports = connectDB;