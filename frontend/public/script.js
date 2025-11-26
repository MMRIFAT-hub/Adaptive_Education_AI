const container = document.getElementById('container');
const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');

// Switch to Sign Up form
signUpButton.addEventListener('click', () => {
  container.classList.add("right-panel-active");
});

// Switch to Sign In form
signInButton.addEventListener('click', () => {
  container.classList.remove("right-panel-active");
});
// Sidebar toggle functionality
const toggleButton = document.getElementById("toggle-sidebar");
const sidebar = document.getElementById("sidebar");

toggleButton.addEventListener("click", () => {
    sidebar.classList.toggle("open"); // Toggle 'open' class to show or hide the sidebar
});
