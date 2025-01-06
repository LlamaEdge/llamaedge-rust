use endpoints::chat::{
    ChatCompletionRequestMessage, ChatCompletionSystemMessage, ChatCompletionUserMessage,
    ChatCompletionUserMessageContent,
};
use llamaedge::{params::ChatParams, Client};

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:8080";

    // Create a client
    let client = Client::new(SERVER_BASE_URL).unwrap();

    // create messages
    let mut messages = Vec::new();
    let system_message = ChatCompletionRequestMessage::System(ChatCompletionSystemMessage::new(
        "You are a helpful assistant. Answer questions as concisely and accurately as possible.",
        None,
    ));
    messages.push(system_message);
    let user_message = ChatCompletionRequestMessage::User(ChatCompletionUserMessage::new(
        ChatCompletionUserMessageContent::Text("What is the capital of France?".to_string()),
        None,
    ));
    messages.push(user_message);

    // send chat completion request
    if let Ok(generation) = client.chat(&messages[..], &ChatParams::default()).await {
        println!("AI response: {}", generation);
    }
}
