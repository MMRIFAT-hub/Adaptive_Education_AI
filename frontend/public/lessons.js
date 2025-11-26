// server.js
import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import OpenAI from 'openai'

dotenv.config()
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

const app = express()
app.use(cors())
app.use(express.json())

app.post('/generateQuestions', async (req, res) => {
  const { topic } = req.body
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are a helpful question generator.' },
        {
          role: 'user',
          content: `Generate 5 multiple-choice questions (with 4 options each, mark the correct answer) for students struggling with the topic: ${topic}.`,
        },
      ],
      temperature: 0.7,
      max_tokens: 800,
    })
    res.json({ questions: response.choices[0].message.content })
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: 'OpenAI error' })
  }
})

const PORT = process.env.PORT || 3000
app.listen(PORT, () => console.log(`Listening on ${PORT}`))
