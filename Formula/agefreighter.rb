class Agefreighter < Formula
  include Language::Python::Virtualenv

  desc "a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
  homepage "https://github.com/rioriost/agefreighter/"
  url "https://files.pythonhosted.org/packages/5c/15/ba89448367a6afde03f0ef3364500462e6713312d1d84c7565cffb373454/agefreighter-1.0.5.tar.gz"
  sha256 "1beaad673d24020eda6e188da7b0e7d1c299484fd877f14f4cc633e8b708d53a"
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
    url "https://files.pythonhosted.org/packages/f6/37/23083fcd6e35492953e8d2aaaa68b860eb422b34627b13f2ce3eb6106061/typing_extensions-4.13.2.tar.gz"
    sha256 "e6c81219bd689f51865d9e372991c540bda33a0379d5573cddb9a3a23f7caaef"
  end

  def install
    virtualenv_install_with_resources
    system libexec/"bin/python", "-m", "pip", "install", "numpy", "psycopg[binary,pool]"
  end

  test do
    system "#{bin}/agefreighter", "--help"
  end
end
