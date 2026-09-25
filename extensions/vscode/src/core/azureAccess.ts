export type AzureAccessState = "ready" | "signedOut" | "accessRequired";

export interface ExistingAzureAccounts<Account> {
  accounts(): Promise<readonly Account[]>;
  session(account: Account, options: { createIfNone: false; silent: boolean }): Promise<boolean>;
}

/** Account presence and extension-specific session access are separate in VS Code. */
export async function existingAzureAccess<Account>(auth: ExistingAzureAccounts<Account>): Promise<AzureAccessState> {
  const accounts = await auth.accounts();
  if (!accounts.length) {
    return "signedOut";
  }
  for (const account of accounts) {
    if (await auth.session(account, { createIfNone: false, silent: true })) {
      return "ready";
    }
  }
  // Queue the standard Accounts-menu permission request. Do not force login or
  // create a second session just because another extension owns the permission.
  for (const account of accounts) {
    if (await auth.session(account, { createIfNone: false, silent: false })) {
      return "ready";
    }
  }
  return "accessRequired";
}

export class AzureAccessError extends Error {
  constructor(public readonly state: Exclude<AzureAccessState, "ready">) {
    super(state === "signedOut"
      ? "No Microsoft account is signed into VS Code. Sign in through Azure Resources, then refresh Azure access."
      : "A Microsoft account is already signed in. Open VS Code's Accounts menu (profile icon), approve the AGEFreighter access request for your Azure account, then refresh Azure access. Azure Resources permission is separate from AGEFreighter permission.");
    this.name = "AzureAccessError";
  }
}
