class Agefreighter < Formula
  include Language::Python::Virtualenv

  desc "a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
  homepage "https://github.com/rioriost/agefreighter/"
  url "https://files.pythonhosted.org/packages/59/e6/77469e4cf0ce4a89a8fda66287b10e20edc15131d0c6dc5ef23cbcaef99e/agefreighter-1.0.14.tar.gz"
  sha256 "14f31db103d1414f1af28f43baf15b8b6230fc6db549032d8a36c03b8409d0ce"
  license "MIT"

  depends_on "python@3.13"

  resource "aiofiles" do
    url "https://files.pythonhosted.org/packages/0b/03/a88171e277e8caa88a4c77808c20ebb04ba74cc4681bf1e9416c862de237/aiofiles-24.1.0.tar.gz"
    sha256 "22a075c9e5a3810f0c2e48f3008c94d68c65d763b9b03857924c99e57355166c"
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
