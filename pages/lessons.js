// server.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.post('/generateQuestions', async (req, res) => {
  const { topic } = req.body || {};

  if (!topic) {
    return res.status(400).json({ error: 'Must provide a topic' });
  }

  console.log(`[generateQuestions] received topic: "${topic}"`);

  try {
    const gptRes = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',      // or gpt-4 if you have access
      temperature: 0.7,
      max_tokens: 800,
      messages: [
        { role: 'system', content: 'You are a helpful question generator.' },
        {
          role: 'user',
          content: `A student is struggling with the topic "${topic}". 
Please generate exactly 5 unique multiple choice questions about this topic. 
Each question should have 4 options labeled (A)–(D), and clearly mark the correct answer.`,
        },
      ],
    });

    const text = gptRes.choices[0].message.content;
    console.log(`[generateQuestions] GPT output:\n${text}\n`);
    res.json({ questions: text });
  } catch (err) {
    console.error('[generateQuestions] OpenAI error:', err);
    res.status(500).json({ error: 'Failed to generate questions' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
