# SmartMirror-Vox
### Voice-Augmented Magic Mirror for Smart Environments

An intelligent, multimodal smart mirror integrating Artificial Intelligence, Computer Vision, Generative AI, and multilingual voice interaction to create a context-aware personal assistant.

---

## 1. Project Overview

SmartMirror-Vox transforms a conventional mirror into an intelligent, interactive system.

Unlike traditional smart mirrors that primarily display static information, SmartMirror-Vox enables users to communicate through voice commands, interact with AI-powered services, analyze their outfits, identify objects, and generate virtual hairstyle previews.

The system combines a web-based interface, Python Flask backend, n8n workflow orchestration, and external AI services.

**Core capabilities**
- Multilingual voice-based conversations.
- AI-powered outfit and appearance analysis.
- Camera-based object identification.
- Generative AI-based virtual hairstyle previews.
- Voice-activated music interaction.
- Jarvis and Lisa conversational personalities.

---

## 2. Project Demonstration

### Live Smart Mirror Interface

![SmartMirror-Vox Live Demo](assets/smartmirror-live-demo.jpg)

*Live interface showing the smart mirror dashboard, camera interaction, voice assistant controls, weather information and news.*

### System Architecture and n8n Workflow

![SmartMirror-Vox Architecture](assets/smartmirror-architecture.jpg)

*Actual n8n workflow and system architecture illustrating intent classification, request routing, AI processing and response delivery.*

---

## 3. Key Features

| Feature | Description |
|---|---|
| Voice Interaction | Hands-free communication through voice commands |
| Multilingual Support | English, Kannada, Telugu, Hindi, Tamil, Malayalam, Marathi and Bengali |
| Conversational AI | AI-powered conversations using Groq and Google Gemini |
| Dual AI Personas | Jarvis and Lisa assistant personalities |
| Outfit Analysis | AI-based clothing and styling analysis |
| Object Identification | Camera-based visual understanding |
| Virtual Hairstyle Generation | AI-generated hairstyle previews |
| Music Interaction | Voice-based music requests |
| Smart Dashboard | Clock, date, weather and news |
| Workflow Automation | n8n-based intent classification and routing |

---

## 4. Technology Stack

**Frontend**
- HTML5
- CSS3
- JavaScript
- Browser Speech and Camera APIs

**Backend**
- Python
- Flask
- Flask-CORS
- Requests
- Pillow

**Artificial Intelligence**
- Groq
- Google Gemini
- Qwen-VL
- Vision-Language Models
- FLUX.1
- Pollinations

**Automation and Integration**
- n8n
- Webhooks
- REST APIs
- JSON-based workflow processing

**Deployment and Version Control**
- Netlify
- Git
- GitHub

---

## 5. System Architecture

The application follows a modular architecture consisting of four major components.

### 5.1 Presentation Layer

The frontend provides the primary user interaction interface.

Responsibilities:
- Displaying the smart mirror dashboard.
- Showing time, date, weather and news.
- Capturing voice commands.
- Accessing camera input.
- Displaying AI responses and generated images.
- Playing audio responses and music.

### 5.2 Backend Layer – Python Flask

The Flask application acts as the backend controller.

Responsibilities:
- Handling frontend requests.
- Managing application communication.
- Maintaining language and assistant preferences.
- Integrating external APIs.
- Supporting audio and music-related operations.

### 5.3 Workflow Orchestration – n8n

The n8n workflow coordinates the processing of user requests.

Workflow stages:

1. **Webhook:** Receives incoming requests.
2. **Normalize and Classify:** Processes input and identifies user intent.
3. **Intent Switch Router:** Selects the appropriate processing branch.
4. **AI Processing:** Handles general queries, vision analysis, image generation and music requests.
5. **Response Processing:** Prepares the result.
6. **Respond to Webhook:** Returns the response to the application.

### 5.4 AI and External Services

- Groq and Google Gemini – Conversational AI.
- Qwen-VL / Gemini Vision – Visual understanding.
- FLUX.1 / Pollinations – Generative image processing.
- Speech-to-Text and Text-to-Speech – Voice interaction.
- External APIs – Weather, news and multimedia services.

---

## 6. Request Processing Workflow

```text
              USER
                |
                v
       VOICE COMMAND INPUT
                |
                v
        SPEECH RECOGNITION
                |
                v
       SMART MIRROR FRONTEND
                |
                v
        PYTHON FLASK SERVER
                |
                v
           n8n WEBHOOK
                |
                v
       NORMALIZE USER INPUT
                |
                v
       INTENT CLASSIFICATION
                |
                v
        INTENT SWITCH ROUTER
                |
       +--------+--------+--------+
       |        |        |        |
       v        v        v        v
    GENERAL   OUTFIT   IMAGE    MUSIC
     QUERY   ANALYSIS GENERATION REQUEST
       |        |        |        |
       +--------+--------+--------+
                |
                v
       RESPONSE PROCESSING
                |
                v
       RESPOND TO WEBHOOK
                |
                v
       SMART MIRROR INTERFACE
                |
          +-----+-----+
          |           |
          v           v
       TEXT       VOICE OUTPUT
       RESPONSE
```

---

## 7. Supported Languages

The system supports eight languages:

| Language | Support |
|---|---|
| English | Yes |
| Kannada | Yes |
| Telugu | Yes |
| Hindi | Yes |
| Tamil | Yes |
| Malayalam | Yes |
| Marathi | Yes |
| Bengali | Yes |

Users can interact with the assistant using Jarvis or Lisa personalities.

---

## 8. Repository Structure

```text
Smart-Mirror/
│
├── .github/
│   └── workflows/
│
├── services/
│
├── assets/
│   ├── smartmirror-live-demo.jpg
│   └── smartmirror-architecture.jpg
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
├── .gitignore
├── start_mirror.bat
│
└── README.md
```

---

## 9. Installation and Execution

### Prerequisites

- Python 3.9 or later.
- Git.
- pip package manager.
- Modern web browser.
- n8n instance.
- Required external AI service credentials.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sharuu123-gif/Smart-Mirror.git
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

**Windows**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Configure the n8n Workflow

1. Start or access your n8n instance.
2. Import `workflow_smartmirror_vox.json`.
3. Configure the required AI service credentials.
4. Configure the webhook endpoint.
5. Ensure that the Flask backend and frontend use the correct endpoint.

### Step 7: Start the Backend

```bash
python app.py
```

Open the local application address configured by Flask in your browser.

Allow camera and microphone permissions when prompted.

**Configuration note:** External AI services require valid credentials and working API endpoints. The n8n workflow must also be configured for the associated automation features.

---

## 10. Python Dependencies

| Package | Purpose |
|---|---|
| Flask | Backend application framework |
| Flask-CORS | Cross-origin request handling |
| Requests | HTTP communication |
| Pillow | Image processing |

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 11. Engineering Highlights

This project demonstrates the integration of:

- **Multimodal AI:** Combining voice, text and visual interaction.
- **Computer Vision:** Camera-based visual understanding.
- **Generative AI:** AI-powered hairstyle previews.
- **Workflow Automation:** Intent-based request routing using n8n.
- **Backend Engineering:** Python Flask application development.
- **API Integration:** Connecting multiple external AI services.
- **Modular Architecture:** Separating frontend, backend, automation and AI services.

---

## 12. Future Enhancements

- IoT-based smart home automation.
- Personalized user profiles.
- Improved multilingual speech recognition.
- Enhanced real-time computer vision.
- Additional AI-powered personal assistance.
- Improved scalability and response performance.

---

## 13. Project Team

**Institution:** Nagarjuna College of Engineering and Technology, Bengaluru

**Department:** Computer Science and Engineering (Artificial Intelligence & Machine Learning)

**Project:** Major Project – Phase 2

**Course Code:** 23CIP77

**Project Guide:** Dr. Niranjan

| Team Member | USN |
|---|---|
| H. Sharath | 1NC23CI022 |
| K. Rishitha | 1NC23CI025 |
| V. Swetha | 1NC23CI061 |
| Koushik Yadav | 1NC23CI067 |

---

<p align="center">

**SmartMirror-Vox**

*Intelligent Interaction. Multimodal Intelligence. Smarter Environments.*

</p>
