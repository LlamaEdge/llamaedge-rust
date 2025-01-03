use llamaedge::Client;

const SERVER_BASE_URL: &str = "http://localhost:8080";

#[tokio::test]
async fn test_model_list() {
    let client = Client::new(SERVER_BASE_URL).unwrap();
    let result = client.models().await;
    assert!(result.is_ok());

    let models = result.unwrap();
    assert!(!models.is_empty());
}
