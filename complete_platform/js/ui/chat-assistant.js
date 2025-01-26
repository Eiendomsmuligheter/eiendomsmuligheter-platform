class ChatAssistant {
    constructor() {
        this.container = document.getElementById('ai-assistant');
        this.messagesContainer = document.getElementById('chatMessages');
        this.input = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendMessage');
        this.minimizeButton = document.getElementById('minimizeChat');
        
        this.isMinimized = false;
        this.context = {
            propertyData: null,
            analysisResults: null
        };

        this.initializeEventListeners();
        this.displayWelcomeMessage();
    }

    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.handleUserInput());

        // Send message on Enter key
        this.input.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                this.handleUserInput();
            }
        });

        // Minimize/maximize chat
        this.minimizeButton.addEventListener('click', () => this.toggleMinimize());

        // Listen for property analysis updates
        window.addEventListener('propertyAnalysisComplete', (event) => {
            this.updateContext(event.detail);
        });
    }

    async handleUserInput() {
        const userInput = this.input.value.trim();
        if (!userInput) return;

        // Clear input
        this.input.value = '';

        // Add user message to chat
        this.addMessage(userInput, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Process user input and get AI response
            const response = await this.processUserInput(userInput);
            
            // Remove typing indicator
            this.hideTypingIndicator();
            
            // Add AI response to chat
            this.addMessage(response, 'ai');
            
        } catch (error) {
            console.error('Error processing user input:', error);
            this.hideTypingIndicator();
            this.addMessage('Beklager, jeg kunne ikke prosessere forespørselen din. Vennligst prøv igjen.', 'ai', 'error');
        }
    }

    async processUserInput(input) {
        const response = await fetch('/api/chat/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input,
                context: this.context
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.response;
    }

    addMessage(content, type, status = 'normal') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type} ${status}`;
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (type === 'ai') {
            // Add AI icon
            const icon = document.createElement('i');
            icon.className = 'fas fa-robot';
            messageContent.appendChild(icon);
        }

        // Add message text
        const text = document.createElement('p');
        text.textContent = content;
        messageContent.appendChild(text);

        messageDiv.appendChild(messageContent);

        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = this.formatTimestamp(new Date());
        messageDiv.appendChild(timestamp);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Animate message entrance
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        requestAnimationFrame(() => {
            messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        `;
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = this.messagesContainer.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    toggleMinimize() {
        this.isMinimized = !this.isMinimized;
        this.container.classList.toggle('minimized', this.isMinimized);
        
        const icon = this.minimizeButton.querySelector('i');
        icon.className = this.isMinimized ? 'fas fa-plus' : 'fas fa-minus';
    }

    updateContext(data) {
        this.context = {
            ...this.context,
            ...data
        };

        // Notify user about new analysis
        if (data.analysisResults) {
            this.addMessage(
                'Jeg har mottatt ny informasjon om eiendommen. Er det noe spesifikt du lurer på?',
                'ai'
            );
        }
    }

    displayWelcomeMessage() {
        const welcomeMessage = `
            Hei! Jeg er din AI-assistent for eiendomsanalyse. 
            Jeg kan hjelpe deg med:
            • Analyse av reguleringsbestemmelser
            • Tolkning av plantegninger
            • Vurdering av utviklingspotensial
            • Estimering av kostnader og verdier
            • Tekniske spørsmål om bygget

            Hva kan jeg hjelpe deg med i dag?
        `;
        
        this.addMessage(welcomeMessage, 'ai');
    }

    formatTimestamp(date) {
        return date.toLocaleTimeString('nb-NO', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    // Handle specific property-related queries
    async handlePropertyQuery(query) {
        if (!this.context.propertyData) {
            return 'Jeg har ikke tilgang til eiendomsdata ennå. Vennligst last inn en eiendom først.';
        }

        try {
            const response = await fetch('/api/chat/property-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    propertyData: this.context.propertyData,
                    analysisResults: this.context.analysisResults
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.response;

        } catch (error) {
            console.error('Error processing property query:', error);
            throw error;
        }
    }

    // Handle regulatory queries
    async handleRegulatoryQuery(query) {
        if (!this.context.propertyData) {
            return 'Jeg har ikke tilgang til reguleringsdata ennå. Vennligst last inn en eiendom først.';
        }

        try {
            const response = await fetch('/api/chat/regulatory-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    propertyData: this.context.propertyData,
                    analysisResults: this.context.analysisResults
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.response;

        } catch (error) {
            console.error('Error processing regulatory query:', error);
            throw error;
        }
    }

    // Handle development potential queries
    async handleDevelopmentQuery(query) {
        if (!this.context.analysisResults) {
            return 'Jeg har ikke tilgang til analysedataene ennå. Vennligst vent til analysen er fullført.';
        }

        try {
            const response = await fetch('/api/chat/development-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    propertyData: this.context.propertyData,
                    analysisResults: this.context.analysisResults
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data.response;

        } catch (error) {
            console.error('Error processing development query:', error);
            throw error;
        }
    }

    // Generate property report
    async generatePropertyReport() {
        if (!this.context.analysisResults) {
            return 'Jeg kan ikke generere en rapport uten analysedataene. Vennligst vent til analysen er fullført.';
        }

        try {
            const response = await fetch('/api/reports/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    propertyData: this.context.propertyData,
                    analysisResults: this.context.analysisResults
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return `
                Jeg har generert en detaljert rapport for eiendommen. 
                Du kan laste ned rapporten her: ${data.reportUrl}
                
                Rapporten inneholder:
                • Eiendomsdetaljer og teknisk tilstand
                • Reguleringsanalyse og begrensninger
                • Utviklingspotensial og anbefalinger
                • Markedsanalyse og verdivurdering
                • Tekniske tegninger og visualiseringer
            `;

        } catch (error) {
            console.error('Error generating property report:', error);
            throw error;
        }
    }
}