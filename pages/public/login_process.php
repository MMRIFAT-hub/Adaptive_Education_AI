<?php
session_start();
$conn = new mysqli("localhost", "root", "", "login_db_1");

if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}

$email = $_POST['email'];
$password = $_POST['password'];

$stmt = $conn->prepare("SELECT * FROM users WHERE email=?");
$stmt->bind_param("s", $email);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows === 1) {
  $user = $result->fetch_assoc();
  if (password_verify($password, $user['password'])) {
    $_SESSION['user'] = $user['name'];
    echo "Login successful. Welcome " . $_SESSION['user'];
    
     header("Location: index.html");
  } else {
    echo "Incorrect password.";
  }
} else {
  echo "No user found with that email.";
}
?>
