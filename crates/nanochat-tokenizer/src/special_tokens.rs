//! Special token handling

/// Special tokens used in nanochat tokenization
///
/// These tokens are used to mark document boundaries, conversation structure,
/// and padding for batching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Beginning of Sequence token - marks the start of a document
    bos: String,
    /// End of Sequence token - marks the end of a document
    eos: String,
    /// Padding token - used for batching sequences of different lengths
    pad: String,
    /// User message start token
    user_start: String,
    /// User message end token
    user_end: String,
    /// Assistant message start token
    assistant_start: String,
    /// Assistant message end token
    assistant_end: String,
    /// Python code block start token
    python_start: String,
    /// Python code block end token
    python_end: String,
    /// Output block start token
    output_start: String,
    /// Output block end token
    output_end: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: "<|bos|>".to_string(),
            eos: "<|eos|>".to_string(),
            pad: "<|pad|>".to_string(),
            user_start: "<|user_start|>".to_string(),
            user_end: "<|user_end|>".to_string(),
            assistant_start: "<|assistant_start|>".to_string(),
            assistant_end: "<|assistant_end|>".to_string(),
            python_start: "<|python_start|>".to_string(),
            python_end: "<|python_end|>".to_string(),
            output_start: "<|output_start|>".to_string(),
            output_end: "<|output_end|>".to_string(),
        }
    }
}

impl SpecialTokens {
    /// Create a new SpecialTokens instance with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the Beginning of Sequence token
    pub fn bos(&self) -> &str {
        &self.bos
    }

    /// Get the End of Sequence token
    pub fn eos(&self) -> &str {
        &self.eos
    }

    /// Get the Padding token
    pub fn pad(&self) -> &str {
        &self.pad
    }

    /// Get the user start token
    pub fn user_start(&self) -> &str {
        &self.user_start
    }

    /// Get the user end token
    pub fn user_end(&self) -> &str {
        &self.user_end
    }

    /// Get the assistant start token
    pub fn assistant_start(&self) -> &str {
        &self.assistant_start
    }

    /// Get the assistant end token
    pub fn assistant_end(&self) -> &str {
        &self.assistant_end
    }

    /// Get the python start token
    pub fn python_start(&self) -> &str {
        &self.python_start
    }

    /// Get the python end token
    pub fn python_end(&self) -> &str {
        &self.python_end
    }

    /// Get the output start token
    pub fn output_start(&self) -> &str {
        &self.output_start
    }

    /// Get the output end token
    pub fn output_end(&self) -> &str {
        &self.output_end
    }

    /// Get all special tokens as a vector
    pub fn all(&self) -> Vec<&str> {
        vec![
            self.bos(),
            self.eos(),
            self.pad(),
            self.user_start(),
            self.user_end(),
            self.assistant_start(),
            self.assistant_end(),
            self.python_start(),
            self.python_end(),
            self.output_start(),
            self.output_end(),
        ]
    }

    /// Check if a token is a special token
    pub fn is_special(&self, token: &str) -> bool {
        self.all().contains(&token)
    }
}
