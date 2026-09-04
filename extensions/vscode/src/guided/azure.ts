import {
  AzureSubscription,
  VSCodeAzureSubscriptionProvider
} from "@microsoft/vscode-azext-azureauth";
import * as vscode from "vscode";
import { parseAzureResourceID, parseResourcePage, ResourceSummary } from "../core/azure";

export interface AzureSubscriptionSummary {
  id: string;
  name: string;
  tenantId: string;
  accountLabel: string;
}

export interface AzurePlacement {
  resourceId: string;
  location: string;
  zone?: string;
  confidence: "verified";
}

export class AzureSession implements vscode.Disposable {
  private readonly provider = new VSCodeAzureSubscriptionProvider();
  private subscriptionsByID = new Map<string, AzureSubscription>();

  public dispose(): void {
    this.provider.dispose();
  }

  public async subscriptions(): Promise<AzureSubscriptionSummary[]> {
    if (!await this.provider.isSignedIn()) {
      throw new Error("Sign in to Azure in VS Code before starting a guided migration.");
    }
    const subscriptions = await this.provider.getSubscriptions(true);
    this.subscriptionsByID = new Map(subscriptions.map((subscription) => [subscription.subscriptionId, subscription]));
    return subscriptions.map((subscription) => ({
      id: subscription.subscriptionId,
      name: subscription.name,
      tenantId: subscription.tenantId,
      accountLabel: subscription.account.label
    }));
  }

  public async placement(subscriptionID: string, resourceID: string): Promise<AzurePlacement> {
    const subscription = await this.subscription(subscriptionID);
    const parsed = parseAzureResourceID(resourceID);
    if (parsed.subscriptionId.toLocaleLowerCase() !== subscriptionID.toLocaleLowerCase()) {
      throw new Error("The source resource ID belongs to a different subscription.");
    }
    const endpoint = subscription.environment.resourceManagerEndpointUrl.replace(/\/$/, "");
    const scope = `${endpoint}/.default`;
    const token = await subscription.credential.getToken(scope);
    if (!token?.token) {
      throw new Error("VS Code could not obtain an Azure Resource Manager token.");
    }
    const url = new URL(`${endpoint}/subscriptions/${encodeURIComponent(subscriptionID)}` +
      `/resourceGroups/${encodeURIComponent(parsed.resourceGroup)}/resources`);
    url.searchParams.set("api-version", "2021-04-01");
    const resources: ResourceSummary[] = [];
    const endpointOrigin = new URL(endpoint).origin;
    let nextURL: URL | undefined = url;
    for (let pageNumber = 0; nextURL && pageNumber < 100; pageNumber += 1) {
      if (nextURL.protocol !== "https:" || nextURL.origin !== endpointOrigin) {
        throw new Error("Azure Resource Manager returned an unsafe continuation link.");
      }
      const response = await fetch(nextURL, {
        headers: { authorization: `Bearer ${token.token}` },
        signal: AbortSignal.timeout(30_000)
      });
      if (!response.ok) {
        throw new Error(`Azure Resource Manager returned ${response.status} while resolving the source resource.`);
      }
      const page = parseResourcePage(await response.json() as unknown);
      resources.push(...page.resources);
      nextURL = page.nextLink ? new URL(page.nextLink) : undefined;
    }
    if (nextURL) {
      throw new Error("Azure Resource Manager resource pagination exceeded the safety limit.");
    }
    const resource = resources.find((item) => item.id.toLocaleLowerCase() === resourceID.toLocaleLowerCase());
    if (!resource) {
      throw new Error("The source resource was not found in the selected subscription and resource group.");
    }
    return {
      resourceId: resource.id,
      location: resource.location,
      zone: resource.zones.length === 1 ? resource.zones[0] : undefined,
      confidence: "verified"
    };
  }

  private async subscription(subscriptionID: string): Promise<AzureSubscription> {
    let subscription = this.subscriptionsByID.get(subscriptionID);
    if (!subscription) {
      await this.subscriptions();
      subscription = this.subscriptionsByID.get(subscriptionID);
    }
    if (!subscription) {
      throw new Error("Select an Azure subscription visible in the VS Code Azure account.");
    }
    return subscription;
  }
}
