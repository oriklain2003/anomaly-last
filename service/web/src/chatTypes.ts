export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface ChatRequest {
    messages: ChatMessage[];
    context?: any; // The analysis result to provide as context
}

export interface ChatResponse {
    message: ChatMessage;
}

