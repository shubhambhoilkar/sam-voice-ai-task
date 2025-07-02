import React, { useState, useEffect, useRef } from 'react';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [listeningTimeout, setListeningTimeout] = useState(null);

  const ws = useRef(null);
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  
  // WebSocket Connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws.current = new WebSocket(`${protocol}://${window.location.hostname}:9900/ws`);

    ws.current.onopen = () => {
      console.log('WebSocket connected.');
      setIsConnected(true);
    };

    ws.current.onmessage = (event) => {
      console.log('Message from server:', event.data);
      try {
        const data = JSON.parse(event.data);
        console.log("🧠 Received base64 length: ",data.audio?.lenght);

        if (data.pause) {
          console.log("Pausing audio due to user interuption.");
          if (audioRef.current){
            audioRef.current.pause();
          }
          return;
        }

        setMessages((prev) => [...prev, { sender: 'bot', text: data.text, audio: data.audio }]);
        playAudio(data.audio);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.current.onclose = () => {
      console.log('WebSocket closed.');
      setIsConnected(false);
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => ws.current && ws.current.close();
  }, []);

  // Speech Recognition Setup
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Speech recognition not supported in this browser.');
      return;
    }

    const SpeechRecognition = window.webkitSpeechRecognition;
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.lang = 'en-US';
    recognitionRef.current.interimResults = false;
    recognitionRef.current.maxAlternatives = 1;
    recognitionRef.current.continuous = false;

    recognitionRef.current.onstart = () => {
      setIsListening(true);
      console.log('Voice recognition started.');
    };

    if (audioRef.current){
      audioRef.current.pause();
    }

    recognitionRef.current.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      console.log('Voice input:', transcript);
      setInput('');

      if (listeningTimeout) clearTimeout(listeningTimeout);

      const timeout = setTimeout(() => {
        sendMessage(transcript);
      }, 500);
      setListeningTimeout(timeout);
    };

    recognitionRef.current.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      restartListening();
    };

    recognitionRef.current.onend = () => {
      console.log('Voice recognition ended.');
      restartListening();
    };

    startListening();

    return () => {
      if (recognitionRef.current) recognitionRef.current.stop();
      if (listeningTimeout) clearTimeout(listeningTimeout);
    };
  }, []);

  const restartListening = () => {
    if (recognitionRef.current) {
      setTimeout(() => {
        recognitionRef.current.start();
      }, 500);
    }
  };

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
    }
  };

  const sendMessage = (textToSend) => {
    const text = textToSend || input.trim();
    if (text === '') return;

    const newMessages = [...messages, { sender: 'user', text }];
    setMessages(newMessages);

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(text);
    } else {
      console.error('WebSocket not connected.');
    }

    setInput('');
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      sendMessage();
    }
  };

  const playAudio = (base64Audio) => {
  if (!base64Audio) return;

  if (audioRef.current) {
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
  }

  const audio = new Audio(`data:audio/mpeg;base64,${base64Audio}`);
  audioRef.current = audio;

  audio.play()
    .then(() => {
      console.log("✅ Audio is playing.");
    })
    .catch((err) => {
      console.error("❌ Audio play error:", err);
    });
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>Sam's Voice Chatbot</h1>
      <p>Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</p>
      <p>Voice Listening: {isListening ? '🎤 Active' : '⏹️ Inactive (Auto-restarting)'}</p>

      <div style={{ border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'scroll' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ margin: '10px 0', textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
            <strong>{msg.sender === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          style={{ width: '60%', padding: '10px' }}
          placeholder="Type your message here"
        />
        <button onClick={() => sendMessage()} style={{ padding: '10px 20px', marginLeft: '10px' }}>
          Send
        </button>
        <button onClick={() => playAudio(messages[messages.length - 1].audio)}>
          🔊 Replay Last Audio
        </button>
      </div>
    </div>
  );
}
