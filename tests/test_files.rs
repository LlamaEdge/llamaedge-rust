use llamaedge::Client;

const SERVER_BASE_URL: &str = "http://localhost:8080";

#[tokio::test]
async fn test_upload_file() {
    let client = Client::new(SERVER_BASE_URL).unwrap();

    let result = client.upload_file("tests/assets/test.wav").await;
    assert!(result.is_ok());
    let file_object = result.unwrap();
    assert_eq!(file_object.filename, "test.wav");
}
