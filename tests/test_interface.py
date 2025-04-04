from verl_agent_env.interface import initialize_environment, close_environment

def test_initialize_close_environment():
    # Test initializing a basic environment
    result = initialize_environment("CartPole-v1", seed=42)
    
    # Check that we get back the expected response structure
    assert isinstance(result, dict)
    assert "message" in result
    assert "env_id" in result 
    assert "observation" in result
    assert "info" in result
    
    # Check the message is as expected
    assert result["message"] == "Environment 'CartPole-v1' initialized successfully."
    
    # Check we got back a valid env_id
    assert isinstance(result["env_id"], str)
    assert len(result["env_id"]) > 0
    
    # Check observation and info are returned
    assert result["observation"] is not None
    assert isinstance(result["info"], dict)

    # Clean up environment 
    close_result = close_environment(result["env_id"])
    assert isinstance(close_result, dict)
    assert "message" in close_result
    assert close_result["message"] == f"Environment with ID '{result['env_id']}' closed successfully."
