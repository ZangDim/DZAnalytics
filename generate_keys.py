import bcrypt

# Sample data
names = ["Dimitris Zanganas", "admin"]
usernames = ["dzagganas", "admin"]
# passwords = ["XXXXXX", "XXXXXX"]

# Open the passwords.txt file in append mode
with open("passwords.txt", "a") as file:
    for username, password in zip(usernames, passwords):
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Convert the hashed password to a string for storage
        hashed_password_str = hashed_password.decode('utf-8')

        # Save the username and hashed password to the file
        file.write(f"{username}:{hashed_password_str}\n")

print("Hashed passwords stored in passwords.txt")
