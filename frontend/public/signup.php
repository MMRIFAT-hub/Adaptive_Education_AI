<?php
$conn = new mysqli("localhost", "root", "", "login_db_1");

if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}

$name = trim($_POST['name']);
$email = trim($_POST['email']);
$passwordRaw = $_POST['password'];
$password = password_hash($passwordRaw, PASSWORD_DEFAULT);

// Validate email
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
  die("Invalid email format.");
}

// Check for existing email
$check = $conn->prepare("SELECT id FROM users WHERE email = ?");
$check->bind_param("s", $email);
$check->execute();
$check->store_result();

if ($check->num_rows > 0) {
  die("Email already registered.");
}

// Insert user
$stmt = $conn->prepare("INSERT INTO users (name, email, password) VALUES (?, ?, ?)");
$stmt->bind_param("sss", $name, $email, $password);

if ($stmt->execute()) {
  header("Location: login_success.html"); // Or dashboard
  exit();
} else {
  echo "Error: " . $stmt->error;
}
?>
