use endpoints::chat::{
    ChatCompletionRequestMessage, ChatCompletionSystemMessage, ChatCompletionUserMessage,
    ChatCompletionUserMessageContent,
};
use llamaedge::{params::RagChatParams, Client};

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
        ChatCompletionUserMessageContent::Text(
            "What is the location of Paris, France along with the Seine River?".to_string(),
        ),
        None,
    ));
    messages.push(user_message);

    // send chat completion request
    if let Ok(res) = client
        .retrieve_rag_context(&messages[..], RagChatParams::default())
        .await
    {
        println!("Retrieved context:\n{:#?}", res);
    }
}
