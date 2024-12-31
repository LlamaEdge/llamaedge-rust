use endpoints::chat::{
    ChatCompletionChunk, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionUserMessage, ChatCompletionUserMessageContent,
};
use futures::StreamExt;
use llamaedge::{params::ChatParams, Client};

const SERVER_BASE_URL: &str = "http://localhost:10086";

#[tokio::test]
async fn test_chat() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

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

    let result = client.chat(&messages[..], &ChatParams::default()).await;

    assert!(result.is_ok());
    let generation = result.unwrap();
    println!("{}", generation);
}

#[tokio::test]
async fn test_chat_stream() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

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

    let result = client
        .chat_stream(&messages[..], &ChatParams::default())
        .await;
    assert!(result.is_ok());
    let mut stream = result.unwrap();

    // iterate over the stream
    let mut output = String::new();
    while let Some(item) = stream.next().await {
        if let Ok(event) = item {
            let event_parts = event
                .split("data: ")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<&str>>();

            for part in event_parts.iter() {
                if *part == "[DONE]" {
                    break;
                }

                if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(part) {
                    if !chunk.choices.is_empty() {
                        if let Some(content) = &chunk.choices[0].delta.content {
                            let content = content.trim();
                            if !content.is_empty() {
                                // append content to output
                                output.push_str(content);
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(!output.is_empty());
    assert!(output.contains("Paris"));
    println!("output: {}", output);
}
