# Fake Review Detection

This Django project is designed to perform text classification, specifically to distinguish between original and computer-generated reviews. The application provides functionalities for user registration, login, data loading, preprocessing, model training, and prediction.

## Features

- **User Registration and Login**: Users can create an account and log in to the application.
- **Data Loading**: Users can upload CSV files containing text data.
- **Data Viewing**: Users can view the uploaded data.
- **Data Preprocessing**: The application preprocesses the text data for model training.
- **Model Training**: Users can train models using K-Nearest Neighbors, Gaussian Naive Bayes, and Logistic Regression.
- **Prediction**: Users can input text to predict whether it is original or computer-generated.

## Project Structure

- **Views**: Contains the logic for handling HTTP requests and rendering HTML templates.
- **Models**: Defines the database structure for user registration.
- **Templates**: HTML files for different pages in the application.
- **Static Files**: CSS, JavaScript, and image files used in the application.

## Setup Instructions

### Prerequisites

- Python 3.x
- Django 3.x or higher
- Pandas
- NLTK
- Scikit-learn

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Swapnanil-Sarkar/Fake-Review-Detection.git
    cd Fake-Review-Detection
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run migrations**:
    ```sh
    python manage.py migrate
    ```

5. **Run the development server**:
    ```sh
    python manage.py runserver
    ```

### Configuration

1. **Database Configuration**:
   - By default, Django uses SQLite. You can change the database settings in `settings.py`.

2. **NLTK Data**:
   - Make sure to download the necessary NLTK data files:
     ```sh
     python -m nltk.downloader stopwords punkt
     ```

### Usage

1. **Home Page**:
   - Access the home page at `http://127.0.0.1:8000/`.

2. **Registration**:
   - Navigate to the registration page to create a new account.

3. **Login**:
   - Log in with your registered credentials.

4. **Load Data**:
   - Upload a CSV file containing text data on the Load Data page.

5. **View Data**:
   - View the uploaded data on the View Data page.

6. **Preprocessing**:
   - Preprocess the uploaded data on the Preprocessing page.

7. **Model Training**:
   - Train a model using the available algorithms on the Model Training page.

8. **Prediction**:
   - Enter text to predict whether it is original or computer-generated on the Prediction page.

## Code Explanation

### Views

- **index**: Renders the home page.
- **about**: Renders the about page.
- **login**: Handles user login functionality.
- **registration**: Handles user registration functionality.
- **userhome**: Renders the user home page after login.
- **load**: Handles CSV file upload and data loading.
- **view**: Displays the uploaded data.
- **preprocessing**: Preprocesses the data for model training.
- **model**: Handles model training using different algorithms.
- **prediction**: Handles text prediction using the trained model.

### Models

- **Register**: Defines the user registration model.

### Templates

- **index.html**: Home page template.
- **about.html**: About page template.
- **login.html**: Login page template.
- **registration.html**: Registration page template.
- **userhome.html**: User home page template.
- **load.html**: Data loading page template.
- **view.html**: Data viewing page template.
- **preprocessing.html**: Data preprocessing page template.
- **model.html**: Model training page template.
- **prediction.html**: Prediction page template.

## Dependencies

- Django
- Pandas
- NLTK
- Scikit-learn

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License.

---

By following these instructions, you should be able to set up and run the project locally. If you encounter any issues or have questions, feel free to open an issue on GitHub.
