const fs = require('fs');

const elements = {};
const documentMock = {
  getElementById: (id) => {
    if (!elements[id]) {
      elements[id] = {
        innerText: '',
        innerHTML: '',
        style: {},
        classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
        addEventListener: () => {},
        setAttribute: () => {},
        getAttribute: () => '',
        play: async () => {},
        pause: () => {},
        getContext: () => ({ drawImage: () => {}, getImageData: () => ({ data: [] }) })
      };
    }
    return elements[id];
  },
  createElement: (tag) => ({
    style: {},
    classList: { add: () => {}, remove: () => {} },
    appendChild: () => {},
    setAttribute: () => {}
  }),
  body: { appendChild: () => {} },
  addEventListener: () => {}
};

const windowMock = {
  addEventListener: () => {},
  document: documentMock,
  location: { href: 'http://127.0.0.1:5500', hostname: '127.0.0.1', port: '5500', protocol: 'http:' },
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  speechSynthesis: { getVoices: () => [], speak: () => {}, cancel: () => {}, onvoiceschanged: null },
  Audio: class { constructor() { this.play = async () => {}; this.pause = () => {}; } },
  Date: Date,
  setInterval: () => {},
  setTimeout: () => {},
  navigator: { mediaDevices: { getUserMedia: async () => ({}) } }
};

global.window = windowMock;
global.document = documentMock;
global.localStorage = windowMock.localStorage;
global.speechSynthesis = windowMock.speechSynthesis;
global.Audio = windowMock.Audio;
global.navigator = windowMock.navigator;

const code = fs.readFileSync('temp_script_2.js', 'utf-8');
try {
  eval(code);
  console.log("SUCCESS! Script evaluated without runtime exception.");
} catch (err) {
  console.error("RUNTIME ERROR during top-level execution:", err);
}
