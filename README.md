# 🪞 SmartMirror-Vox
### Voice-Augmented Magic Mirror for Smart Environments

**SmartMirror-Vox** is an AI-powered smart mirror that transforms a traditional mirror into an intelligent, interactive personal assistant using Artificial Intelligence, Computer Vision, Voice Recognition, and IoT technologies.

The system combines a web-based smart mirror interface, Python Flask backend, and n8n workflow automation to provide multilingual voice conversations, outfit analysis, object identification, virtual hairstyle previews, and multimedia entertainment.

## 🚀 Project Overview

Traditional smart mirrors mainly display static information such as time, date, and weather. SmartMirror-Vox extends this functionality by introducing voice-based interaction and AI-powered visual intelligence.

Users can interact with the mirror through voice commands, receive personalized responses, analyze their outfits, identify objects, and generate virtual hairstyle previews.

## ✨ Key Features

- **AI Voice Assistant:** Hands-free interaction using voice commands.
- **Multilingual Support:** Supports English, Kannada, Telugu, Hindi, Tamil, Malayalam, Marathi, and Bengali.
- **Dual AI Personas:** Choose between Jarvis and Lisa conversational personalities.
- **AI Outfit Analysis:** Analyze clothing combinations, colors, and styling.
- **Object Identification:** Identify objects using camera input and AI vision.
- **Virtual Hairstyle Try-On:** Generate hairstyle previews using generative AI.
- **Smart Dashboard:** Displays time, date, weather, and news.
- **Voice-Activated Music:** Request and play music using voice commands.
- **Workflow Automation:** Uses n8n to classify requests and route them to appropriate AI services.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | HTML5, CSS3, JavaScript |
| Backend | Flask |
| API Communication | REST APIs, Flask-CORS, Requests |
| Workflow Automation | n8n |
| Conversational AI | Groq, Google Gemini |
| Computer Vision | Qwen-VL / Gemini Vision |
| Image Generation | FLUX.1 / Pollinations |
| Image Processing | Pillow |
| Voice Interaction | Speech-to-Text (STT), Text-to-Speech (TTS) |
| Hardware | Monitor, Two-Way Mirror, Webcam, Microphone, Speakers |

## 🏗️ System Architecture

The system consists of three major layers:

**1. Frontend Layer**
- Developed using HTML, CSS, and JavaScript.
- Displays the smart mirror dashboard.
- Handles microphone and camera interaction.
- Displays AI-generated responses and visual results.

**2. Backend Layer**
- Developed using Python Flask.
- Manages user sessions and language preferences.
- Handles API communication and audio streaming.
- Connects the frontend with the automation workflow.

**3. AI and Automation Layer**
- Uses n8n for workflow orchestration.
- Uses Groq and Gemini for conversational AI.
- Uses vision models for outfit and object analysis.
- Uses generative AI for hairstyle previews.

## 🔄 Working Methodology

1. The user gives a voice command through the mirror.
2. Speech recognition converts the voice into text.
3. The request is forwarded to the n8n webhook.
4. The workflow identifies the user's intent.
5. The request is routed to the appropriate AI service.
6. The result is returned to the Flask backend.
7. The mirror displays the response and provides voice output.

## 📂 Project Structure

```text
Smart-Mirror/
│
├── .github/
│   └── workflows/
│
├── services/
│
├── app.py
├── index.html
├── mirror.html
├── requirements.txt
├── workflow_smartmirror_vox.json
│
├── netlify.toml
├── _headers
├── _redirects
├── start_mirror.bat
│
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.9 or above
- pip package manager
- Git
- Web browser
- n8n account or running n8n instance
- Required API credentials for external AI services

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Navigate to the Project

```bash
cd Smart-Mirror
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Configure External Services

Configure the required API credentials and n8n webhook URL according to the application's configuration.

Do not upload API keys or passwords to GitHub.

### Step 7: Run the Application

```bash
python app.py
```

Open the local application URL configured by Flask in your browser.

## 📸 Project Screenshots

The following screenshots demonstrate the SmartMirror-Vox interface and its AI capabilities.

Add your actual project screenshots here.

```markdown
![Smart Mirror Dashboard](screenshots/dashboard.png)

![AI Vision Analysis](screenshots/vision-analysis.png)

![AI Hairstyle Generation](screenshots/hairstyle-preview.png)

![n8n Workflow](screenshots/n8n-workflow.png)
```

## 🎯 Project Objectives

- Develop an interactive AI-powered smart mirror.
- Enable multilingual voice-based communication.
- Integrate computer vision for outfit and object analysis.
- Implement generative AI for virtual hairstyle previews.
- Automate user requests using n8n workflows.
- Create a modular architecture for future smart environment integration.

## 🔮 Future Enhancements

- Advanced IoT home automation integration.
- Personalized user profiles and preferences.
- Improved AI vision capabilities.
- Additional regional language support.
- Enhanced real-time interaction and response optimization.
- Integration with additional smart home devices.

## 👨‍💻 Project Team

**Department:** Computer Science and Engineering (Artificial Intelligence and Machine Learning)

**Institution:** Nagarjuna College of Engineering and Technology, Bengaluru

**Project Guide:** Dr. Niranjan

| Team Member | USN |
|---|---|
| H. Sharath | 1NC23CI022 |
| K. Rishitha | 1NC23CI025 |
| V. Swetha | 1NC23CI061 |
| Koushik Yadav | 1NC23CI067 |
