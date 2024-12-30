# Rust API for LlamaEdge

> [!IMPORTANT]
> This project is still in the early stages of development. The API is not stable and may change.

## Usage

Add the following to your `Cargo.toml`:

```toml
[dependencies]
llamaedge = "0.0.1"
endpoints = "0.23.0"
tokio = { version = "1.42.0", features = ["full"] }
```

Then, you can use the following code to send a chat completion request:

```rust
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
        println!("assistant:{}", generation);
    }
}
```

**Note:** To run the example, LlamaEdge API server should be deployed and running on your local machine. Refer to [Quick Start](https://github.com/LlamaEdge/LlamaEdge?tab=readme-ov-file#quick-start) for more details on how to deploy and run the server.
