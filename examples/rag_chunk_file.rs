use llamaedge::Client;

#[tokio::main]
async fn main() {
    const SERVER_BASE_URL: &str = "http://localhost:10086";

    let client = Client::new(SERVER_BASE_URL).unwrap();

    let file_path = "tests/assets/paris.txt";
    match client.rag_chunk_file(file_path, 1024).await {
        Ok(chunks_response) => println!("{:#?}", chunks_response),
        Err(e) => println!("Error: {}", e),
    }
}
