#[cfg(feature = "image")]
mod tests {
    use llamaedge::{params::ImageCreateParams, Client};

    const SERVER_BASE_URL: &str = "http://localhost:8080";

    #[tokio::test]
    async fn test_image_create() {
        let client = Client::new(SERVER_BASE_URL).unwrap();

        let result = client
            .create_image("A lovely dog", ImageCreateParams::default())
            .await;
        assert!(result.is_ok());

        let image = result.unwrap();
        assert!(image.len() > 0);
    }
}
