class Agefreighter < Formula
  include Language::Python::Virtualenv

  desc "a Python package that helps you to create a graph database using Azure Database for PostgreSQL."
  homepage "https://github.com/rioriost/agefreighter/"
  url "https://files.pythonhosted.org/packages/cc/63/3c8f14003f438ee59de57d5c4668fea97520cc04ef489479bf0fa5f7b163/agefreighter-1.0.4.tar.gz"
  sha256 "ad4b4883ebd4e40346f56806a383fe9f70d8970dd81d6e2f7b968c0a81c79750"
  license "MIT"

  depends_on "python@3.13"

  resource "aiofiles" do
    url "https://files.pythonhosted.org/packages/0b/03/a88171e277e8caa88a4c77808c20ebb04ba74cc4681bf1e9416c862de237/aiofiles-24.1.0.tar.gz"
    sha256 "22a075c9e5a3810f0c2e48f3008c94d68c65d763b9b03857924c99e57355166c"
  end

  resource "shtab" do
    url "https://files.pythonhosted.org/packages/a9/e4/13bf30c7c30ab86a7bc4104b1c943ff2f56c1a07c6d82a71ad034bcef1dc/shtab-1.7.1.tar.gz"
    sha256 "4e4bcb02eeb82ec45920a5d0add92eac9c9b63b2804c9196c1f1fdc2d039243c"
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
    system "#{bin}/agefreighter", "--help"
  end
end
