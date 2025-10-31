class Agefreighter < Formula
  include Language::Python::Virtualenv

  desc "a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
  homepage "https://github.com/rioriost/agefreighter/"
  url "https://files.pythonhosted.org/packages/15/11/04078a86da72ac1106b61fde6aa88dd4843732b043652a20bb615859b15c/agefreighter-1.0.19.tar.gz"
  sha256 "ec2cd4f2b415d5378c80ef7fe7f64bedd50641fa5f0f7257a3eddf79711b5adc"
  license "MIT"

  depends_on "python@3.13"

  resource "aiofiles" do
    url "https://files.pythonhosted.org/packages/41/c3/534eac40372d8ee36ef40df62ec129bee4fdb5ad9706e58a29be53b2c970/aiofiles-25.1.0.tar.gz"
    sha256 "a8d728f0a29de45dc521f18f07297428d56992a742f0cd2701ba86e44d23d5b2"
  end

  resource "shtab" do
    url "https://files.pythonhosted.org/packages/5a/3e/837067b970c1d2ffa936c72f384a63fdec4e186b74da781e921354a94024/shtab-1.7.2.tar.gz"
    sha256 "8c16673ade76a2d42417f03e57acf239bfb5968e842204c17990cae357d07d6f"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/72/94/1a15dd82efb362ac84269196e94cf00f187f7ed21c242792a923cdb1c61f/typing_extensions-4.15.0.tar.gz"
    sha256 "0cea48d173cc12fa28ecabc3b837ea3cf6f38c6d1136f85cbaaf598984861466"
  end

  def install
    virtualenv_install_with_resources
    system libexec/"bin/python", "-m", "pip", "install", "numpy", "psycopg[binary,pool]"
  end

  test do
    system "#{bin}/agefreighter", "--help"
  end
end
