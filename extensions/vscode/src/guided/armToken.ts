interface ArmSession { accessToken: string; account: { id: string } }

/** azureauth's subscription credential captures its original session. Obtain a
 * fresh, silently refreshed session for each ARM request instead. The caller
 * binds this callback to the selected subscription's tenant and account. */
export async function armToken(accountId: string, endpoint: string,
  session: (scopes: string[]) => PromiseLike<ArmSession | undefined | null>): Promise<string> {
  const current = await session([`${endpoint.replace(/\/$/, "")}/.default`]);
  if (!current?.accessToken || current.account.id !== accountId) {
    throw new Error("Azure Resource Manager access for the selected VS Code account is unavailable. Refresh the existing login; no alternate account or automatic request replay was used.");
  }
  return current.accessToken;
}
