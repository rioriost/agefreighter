class Agefreighter < Formula
  include Language::Python::Virtualenv

  desc "a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
  homepage "https://github.com/rioriost/agefreighter/"
  url "https://files.pythonhosted.org/packages/c5/a0/6980fc233e9b93cb13a156cd368f1e2d3d410f6a00502e122b362e1ead0c/agefreighter-1.0.0a5.tar.gz"
  sha256 "d26268b24f0456dcbd7494301ad7a3179e031074a3a13452a8fa6fadd0cfb70e"
  license "MIT"

  depends_on "python@3.13"

  resource "aiofiles" do
    url "https://files.pythonhosted.org/packages/0b/03/a88171e277e8caa88a4c77808c20ebb04ba74cc4681bf1e9416c862de237/aiofiles-24.1.0.tar.gz"
    sha256 "22a075c9e5a3810f0c2e48f3008c94d68c65d763b9b03857924c99e57355166c"
  end

  resource "argcomplete" do
    url "https://files.pythonhosted.org/packages/ee/be/29abccb5d9f61a92886a2fba2ac22bf74326b5c4f55d36d0a56094630589/argcomplete-3.6.0.tar.gz"
    sha256 "2e4e42ec0ba2fff54b0d244d0b1623e86057673e57bafe72dda59c64bd5dee8b"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/df/db/f35a00659bc03fec321ba8bce9420de607a1d37f8342eee1863174c69557/typing_extensions-4.12.2.tar.gz"
    sha256 "1a7ead55c7e559dd4dee8856e3a88b41225abfe1ce8df57b7c13915fe121ffb8"
  end

  def install
    virtualenv_install_with_resources
    system libexec/"bin/python", "-m", "pip", "install", "numpy", "psycopg[binary,pool]"
  end

  test do
    system "#{bin}/phorganize", "--help"
  end
end
