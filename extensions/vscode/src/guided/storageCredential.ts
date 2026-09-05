import type { TokenCredential } from "@azure/core-auth";

export const storageScope = "https://storage.azure.com/.default";
interface StorageSession { accessToken: string; account: { id: string } }

/** AzureSubscription.credential is ARM-only in azureauth: its getToken ignores
 * scopes. Ask the existing account for a Storage session; never fall back to ARM,
 * shared keys or another account. Fetch afresh to let VS Code refresh expiry. */
export function storageCredential(accountId: string,
  session: (scopes: string[]) => PromiseLike<StorageSession | undefined | null>): TokenCredential {
  return { getToken: async scopes => {
    const requested = typeof scopes === "string" ? [scopes] : scopes;
    if (requested.length !== 1 || requested[0] !== storageScope) throw new Error("Only the Azure Storage audience is allowed for file transfer.");
    const current = await session([storageScope]);
    if (!current?.accessToken || current.account.id !== accountId) throw new Error("Azure Storage access for the selected VS Code account is unavailable. The ARM login alone is insufficient; no alternate account or shared key was used.");
    return { token: current.accessToken, expiresOnTimestamp: 0 };
  } };
}
