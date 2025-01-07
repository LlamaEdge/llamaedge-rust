use llamaedge::{params::EmbeddingsParams, Client};

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:8080";

    let client = Client::new(SERVER_BASE_URL).unwrap();

    match client
        .embeddings("Hello, world!".into(), EmbeddingsParams::default())
        .await
    {
        Ok(embeddings) => println!("{:#?}", embeddings),
        Err(e) => println!("Error: {}", e),
    }
}
