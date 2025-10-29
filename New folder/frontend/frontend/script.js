// Get the "Start Quiz" button and the container where MCQs will be displayed
const quizButton = document.getElementById('quiz-button');
const mcqContainer = document.getElementById('mcq-container');

let selectedQuestions = [];  // Store the fetched questions globally

// Add an event listener to the quiz button
quizButton.addEventListener('click', () => {
  const level = document.querySelector('input[name="difficulty"]:checked').value;  // Get selected difficulty
  fetchQuestion(level);  // Call the function to fetch the MCQ from the Flask backend
});

// Function to fetch the MCQ question from Flask backend
function fetchQuestion(level) {
  fetch("http://127.0.0.1:5000/generate_questions", {
    method: "POST",  // POST method
    headers: {
      "Content-Type": "application/json"  // Set content type to JSON
    },
    body: JSON.stringify({
      topic: "Principles of OOP",  // Example topic (you can modify this)
      level: level  // Send the selected difficulty level to the backend
    })
  })
    .then(response => response.json())  // Parse the response as JSON
    .then(data => {
      // Clear previous MCQs
      mcqContainer.innerHTML = ''; 

      // Store the fetched questions globally
      selectedQuestions = data.MCQ;

      // Display the MCQ questions dynamically
      selectedQuestions.forEach((question, index) => {
        const questionElement = document.createElement('div');
        questionElement.classList.add('question');

        // Create question text
        const questionText = document.createElement('p');
        questionText.textContent = `${index + 1}. ${question.question}`;
        questionElement.appendChild(questionText);

        // Create options for each question
        question.options.forEach(option => {
          const optionElement = document.createElement('div');
          optionElement.classList.add('option');

          const inputElement = document.createElement('input');
          inputElement.type = 'radio';
          inputElement.name = `question${index}`;  // Ensure radio buttons are grouped by question
          inputElement.value = option;

          const labelElement = document.createElement('label');
          labelElement.textContent = option;

          optionElement.appendChild(inputElement);
          optionElement.appendChild(labelElement);
          questionElement.appendChild(optionElement);
        });

        // Append the question element to the container
        mcqContainer.appendChild(questionElement);
      });

      // Show the "Submit Quiz" button after loading the questions
      document.getElementById('submit-button').style.display = 'block';
    })
    .catch(error => {
      console.error("Error:", error);
    });
}

// Function to collect answers and display score when the quiz is submitted
function submitQuiz() {
  let score = 0;

  // Loop through each question and check the selected answer
  selectedQuestions.forEach((question, index) => {
    const selectedOption = document.querySelector(`input[name="question${index}"]:checked`);

    // If an option is selected and it matches the correct answer, increment the score
    if (selectedOption && selectedOption.value === question.answer) {
      score++;
    }
  });

  // Display the score in the result container
  document.getElementById('score').textContent = `Score: ${score} / ${selectedQuestions.length}`;
  document.getElementById('result-container').style.display = 'block';  // Display the results
}
