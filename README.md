# Cognitive Load Typing Tracker

A machine learning-powered web application that analyzes typing patterns in real-time to estimate cognitive load levels. The system tracks various typing metrics (speed, pauses, backspaces, bursts) and uses a trained Gradient Boosting Regressor model to predict mental workload during typing tasks.

## Features

### 1. Real-time Typing Analysis
- **Live Typing Test**: Type predefined sentences while the app tracks your behavior
- **Visual Feedback**: Real-time character-by-character feedback (correct/incorrect/current)
- **Accuracy Metrics**: Track accuracy, correct/incorrect characters, and progress
- **Backspace Timeline**: Visual timeline showing when corrections were made

### 2. Cognitive Load Prediction
- **ML-Powered Predictions**: Uses trained Gradient Boosting model for accurate cognitive load estimation
- **Three Load Levels**: Low, Medium, High cognitive load classification
- **Confidence Score**: Behavior-aware confidence metric (40-95%)
- **Smoothed Predictions**: Rolling buffer to reduce noise and improve stability

### 3. Mental Health Assessment
- **Anxiety Quiz**: 10-question assessment with percentage scoring
- **Depression Quiz**: 10-question PHQ-style assessment
- **Stress Quiz**: 10-question perceived stress scale
- **Interactive Chat Interface**: Conversational UI for mental wellness check-ins
- **Wellness Dashboard**: View all assessment results in one place

### 4. Dataset Collection & Export
- **Label Your Sessions**: Mark typing tests as low/medium/high cognitive load
- **Session History**: Browse and review past typing sessions
- **Export Dataset**: Download collected data as CSV for model retraining
- **Persistent Storage**: Local browser storage for history and preferences

## Technology Stack

### Backend
- **Flask**: Python web framework for API endpoints
- **scikit-learn**: Machine learning library (Gradient Boosting Regressor)
- **pandas**: Data manipulation and feature engineering
- **joblib**: Model serialization and loading
- **Flask-CORS**: Cross-origin resource sharing support

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **HTML5/CSS3**: Modern, responsive UI with dark/light theme
- **LocalStorage**: Client-side data persistence
- **Fetch API**: RESTful API communication

### Machine Learning
- **Model**: GradientBoostingRegressor with 50 estimators
- **Features**: 10 engineered features (speed, intervals, pauses, corrections, bursts)
- **Preprocessing**: StandardScaler for feature normalization
- **Validation**: 5-fold cross-validation for model evaluation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup Instructions

1. **Clone or extract the project**
   ```bash
   cd CTTCProject
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Alternatively, install manually:
   ```bash
   pip install flask flask-cors scikit-learn pandas numpy joblib
   ```

3. **Verify model files exist**
   The following pickle files should be present:
   - `cognitive_load_model.pkl` - Trained ML model
   - `scaler.pkl` - Feature scaler
   - `feature_columns.pkl` - Feature order reference

4. **Start the Flask API server**
   ```bash
   python app.py
   ```
   
   The API will start on `http://127.0.0.1:5000`

5. **Open the web interface**
   Open `index.html` in your web browser (you can double-click the file or use a local server)

## Usage Guide

### Typing Test

1. **Start Test**: Open `index.html` in your browser
2. **Type Sentence**: Type the displayed sentence in the text area
3. **View Real-time Stats**: Monitor your typing speed, pauses, backspaces, and cognitive load prediction
4. **Complete Test**: Click "Complete Test" when finished
5. **Label Session**: Choose a cognitive load label (low/medium/high) for dataset collection

### Mental Health Assessment

1. **Access Mental Test**: Click "Mental Test" button from main page
2. **Choose Assessment**: Select Anxiety, Depression, Stress, or "Just fine"
3. **Answer Questions**: Respond to 10 questions for each assessment type
4. **View Results**: See your percentage scores and severity levels
5. **Wellness Dashboard**: Click "View Wellness Dashboard" to see all results

### Dataset Management

1. **View History**: See all past typing sessions in the history panel
2. **Review Sessions**: Click on history items to replay typing patterns
3. **Export Data**: Click "Export Dataset as CSV" to download collected data
4. **Clear Data**: Use "Clear All History" to reset dataset (with confirmation)

### Theme Customization

- Click the "Toggle Theme" button to switch between dark and light modes
- Theme preference is automatically saved to localStorage

## API Endpoints

### POST /predict

Predicts cognitive load based on typing metrics.

**Request Body:**
```json
{
  "speed": 4.5,
  "avgInterval": 150,
  "pause300": 3,
  "pause500": 1,
  "backspaceCount": 5,
  "maxBurst": 12,
  "editRatio": 0.15,
  "duration": 25.4
}
```

**Response:**
```json
{
  "score": 45.23,
  "level": "medium",
  "confidence": 78.5
}
```

**Cognitive Load Levels:**
- **Low**: score < 35 (Minimal mental effort, comfortable typing)
- **Medium**: 35 ≤ score < 65 (Moderate cognitive demand)
- **High**: score ≥ 65 (High mental effort, struggling to type)

## Model Training

### Retrain the Model

To retrain the model with new data:

1. **Update dataset.csv** with your collected data
   - Required columns: `speed`, `avgInterval`, `pause300`, `pause500`, `backspaceCount`, `maxBurst`, `editRatio`, `duration`, `label`

2. **Run training script**
   ```bash
   python train_model.py
   ```

3. **Review metrics**
   - Regression: MAE, R² score
   - Classification: Accuracy, F1 score, confusion matrix
   - Cross-validation: 5-fold R² scores

4. **Updated models** will be saved:
   - `cognitive_load_model.pkl`
   - `scaler.pkl`
   - `feature_columns.pkl`

### Feature Engineering

The model uses 10 engineered features:

| Feature | Description |
|---------|-------------|
| `speed` | Characters per second |
| `avgInterval` | Average time between keystrokes (ms) |
| `pause300` | Number of pauses > 300ms |
| `pause500` | Number of pauses > 500ms |
| `backspaceCount` | Total backspace/delete presses |
| `maxBurst` | Longest continuous typing burst |
| `editRatio` | Ratio of edits to total characters |
| `duration` | Total typing duration (seconds) |
| `pause_ratio_300` | pause300 / total_keys |
| `correction_rate` | backspaceCount / total_keys |

## Project Structure

```
Project/
├── templates/                     # HTML templates for Flask
│   ├── index.html                 # Main typing tracker UI
│   ├── mental.html                # Mental health page
│   └── wellness.html              # Wellness dashboard UI
├── app.py                         # Flask API server
├── train_model.py                 # ML model training script
├── dataset.csv                    # Training dataset
├── cognitive_load_model.pkl       # Trained cognitive load model
├── scaler.pkl                     # Feature scaler
├── feature_columns.pkl            # Feature order reference
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Keyboard Shortcuts

### Main Typing Interface
- **Enter**: Submit typing input
- **Esc**: Reset test
- **?**: Show keyboard shortcuts help

## Browser Compatibility

- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

## Data Privacy

- All data is stored locally in your browser's localStorage
- No data is sent to external servers except the local Flask API
- Mental health assessments are completely private
- Export/delete your data at any time

## Troubleshooting

### API Connection Issues
- Ensure Flask server is running on port 5000
- Check console for CORS errors
- Verify `flask-cors` is installed

### Model Not Loading
- Confirm all `.pkl` files exist in the project directory
- Check Python version compatibility (3.7+)
- Verify scikit-learn version matches training environment

### Predictions Seem Inaccurate
- Collect more labeled data and retrain
- Ensure consistent typing behavior during labeling
- Check feature engineering matches training

### LocalStorage Full
- Export dataset and clear history
- Browser storage limits vary (typically 5-10MB)

## Future Enhancements

- [ ] Real-time stress monitoring with webcam (facial expressions)
- [ ] Multi-language support for typing tests
- [ ] Cloud storage integration for cross-device sync
- [ ] Advanced analytics dashboard with charts
- [ ] Mobile app version
- [ ] Calibration mode for personalized predictions
- [ ] Integration with productivity tools

## Contributing

Contributions are welcome! Areas for improvement:
- Additional typing test sentences and difficulty levels
- Enhanced mental health assessment questions
- Improved ML models (LSTM, Transformer architectures)
- UI/UX enhancements
- Accessibility improvements

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Typing pattern analysis inspired by cognitive psychology research
- Mental health assessments based on GAD-7, PHQ-9, and PSS-10 scales
- Machine learning approach adapted from keystroke dynamics literature

## Contact

For questions, issues, or suggestions, please create an issue in the project repository.

---

**Note**: This tool is for research and educational purposes only. Mental health assessments are not a substitute for professional psychological evaluation or treatment. If you're experiencing mental health concerns, please consult a qualified healthcare professional.
