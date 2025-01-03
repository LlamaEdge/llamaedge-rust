use endpoints::embeddings::InputText;
use llamaedge::{params::EmbeddingsParams, Client};

const SERVER_BASE_URL: &str = "http://localhost:8080";

#[tokio::test]
async fn test_embeddings() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

    let chunks = vec!["Hello, world!", "This is a test."];
    let input = InputText::from(chunks);

    let result = client.embeddings(input, EmbeddingsParams::default()).await;
    assert!(result.is_ok());

    let embeddings = result.unwrap();
    println!("length of embeddings: {}", embeddings.data.len());
    assert!(embeddings.data.len() > 0);
}
