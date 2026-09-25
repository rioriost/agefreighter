import { chmod, lstat, mkdir } from "node:fs/promises";
import { isAbsolute, join, parse } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

// Fixed script, not a command assembled from a path or a webview value. Windows
// does not implement POSIX group/other permission bits; set and inspect its DACL.
const aclScript = `$ErrorActionPreference = 'Stop'
$path = $env:AGEFREIGHTER_PRIVATE_PATH
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = Microsoft.PowerShell.Security\\Get-Acl -LiteralPath $path
if ($env:AGEFREIGHTER_PRIVATE_MODE -eq 'secure') {
  $acl.SetAccessRuleProtection($true, $false)
  foreach ($rule in @($acl.Access)) { $acl.RemoveAccessRuleSpecific($rule) }
  $rule = [System.Security.AccessControl.FileSystemAccessRule]::new($sid, [System.Security.AccessControl.FileSystemRights]::FullControl, [System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit', [System.Security.AccessControl.PropagationFlags]::None, [System.Security.AccessControl.AccessControlType]::Allow)
  $acl.AddAccessRule($rule)
  Microsoft.PowerShell.Security\\Set-Acl -LiteralPath $path -AclObject $acl
  $acl = Microsoft.PowerShell.Security\\Get-Acl -LiteralPath $path
  if (!$acl.AreAccessRulesProtected) { throw 'Directory inheritance is not protected' }
}
$owned = $false
foreach ($rule in @($acl.Access)) {
  $principal = $rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
  if ($rule.AccessControlType -eq 'Allow') {
    if ($principal -ne $sid.Value) { throw 'Unexpected permitted principal' }
    if (($rule.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::FullControl) -eq [System.Security.AccessControl.FileSystemRights]::FullControl) { $owned = $true }
  }
}
if (!$owned) { throw 'Current user lacks private full control' }
`;

async function windowsACL(path: string, mode: "secure" | "check"): Promise<void> {
  const systemRoot = process.env.SystemRoot;
  if (!systemRoot || !isAbsolute(systemRoot)) throw new Error("Windows system tools are unavailable; private storage is blocked.");
  try {
    await promisify(execFile)(join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
      ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(aclScript, "utf16le").toString("base64")],
      { windowsHide: true, timeout: 15000, maxBuffer: 4096, env: { SystemRoot: systemRoot, WINDIR: systemRoot, TEMP: process.env.TEMP, TMP: process.env.TMP, AGEFREIGHTER_PRIVATE_PATH: path, AGEFREIGHTER_PRIVATE_MODE: mode } });
  } catch { throw new Error("Windows private-storage ACL could not be verified. No workflow write is allowed."); }
}

export async function preparePrivateDirectory(root: string): Promise<void> {
  if (!isAbsolute(root) || parse(root).root === root) throw new Error("Private storage requires a dedicated absolute directory.");
  await mkdir(root, { recursive: true, mode: 0o700 });
  const info = await lstat(root);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Private storage cannot use a linked directory.");
  if (process.platform === "win32") await windowsACL(root, "secure");
  else await chmod(root, 0o700);
}

/** Used by the Windows integration test to verify the inherited file ACL. */
export async function verifyPrivateWindowsPath(path: string): Promise<void> {
  if (process.platform !== "win32") throw new Error("Windows ACL validation requires Windows.");
  await windowsACL(path, "check");
}
