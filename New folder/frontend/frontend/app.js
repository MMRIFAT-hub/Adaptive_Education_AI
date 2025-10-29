const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3000;

// Middleware to parse JSON data
app.use(express.json());

// Gemini API endpoint and key (use your actual Gemini API key here)
const GEMINI_API_URL = 'https://api.gemini.com/v1/endpoint'; // Placeholder URL, replace with actual URL
const GEMINI_API_KEY = 'AIzaSyBDut5hD8qRKEpgfcro5pJ-TFCZcAu6ISE';

// Endpoint to call Gemini API
app.post('/generate', async (req, res) => {
    const { topic } = req.body;

    try {
        // Make a request to Gemini API
        const response = await axios.post(GEMINI_API_URL, {
            headers: {
                'Authorization': `Bearer ${GEMINI_API_KEY}`,
                'Content-Type': 'application/json',
            },
            data: {
                topic: topic
            }
        });

        // Extract data from response
        const generatedData = response.data; // Process as per Gemini's response format
        res.json({ question: generatedData.question, answer: generatedData.answer });

    } catch (error) {
        console.error('Error calling Gemini API:', error);
        res.status(500).json({ error: 'Failed to generate question and answer' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
