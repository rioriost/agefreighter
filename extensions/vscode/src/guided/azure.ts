import {
  AzureSubscription,
  getConfiguredAuthProviderId,
  getSessionFromVSCode,
  VSCodeAzureSubscriptionProvider
} from "@microsoft/vscode-azext-azureauth";
import * as vscode from "vscode";
import { AzureAccessError, existingAzureAccess } from "../core/azureAccess";
import {
  AzureLocationSummary,
  parseAzureResourceID,
  parseLocations,
  parseResourcePage,
  ResourceSummary
} from "../core/azure";
import {
  ComputeSkuCapability,
  parseComputeSkus,
  parsePostgresCapabilities,
  parseQuotaUsages,
  parseRetailRates,
  PostgresCapabilities,
  QuotaUsage,
  RetailRate
} from "../core/proposal";

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

export interface AzureRecommendationData {
  postgres: PostgresCapabilities;
  computeSkus: ComputeSkuCapability[];
  postgresQuota: QuotaUsage[];
  computeQuota: QuotaUsage[];
}

export class AzureSession implements vscode.Disposable {
  private readonly provider = new VSCodeAzureSubscriptionProvider();
  private subscriptionsByID = new Map<string, AzureSubscription>();

  public dispose(): void {
    this.provider.dispose();
  }

  public async subscriptions(): Promise<AzureSubscriptionSummary[]> {
    this.subscriptionsByID.clear();
    const access = await existingAzureAccess({
      accounts: async () => vscode.authentication.getAccounts(getConfiguredAuthProviderId()),
      session: async (account, options) => !!await getSessionFromVSCode([], undefined, { ...options, account })
    });
    if (access !== "ready") {
      throw new AzureAccessError(access);
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
    const url = new URL(`${endpoint}/subscriptions/${encodeURIComponent(subscriptionID)}` +
      `/resourceGroups/${encodeURIComponent(parsed.resourceGroup)}/resources`);
    url.searchParams.set("api-version", "2021-04-01");
    const resources = await this.armResourcePages(subscription, url);
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

  public async locations(subscriptionID: string): Promise<AzureLocationSummary[]> {
    const subscription = await this.subscription(subscriptionID);
    const endpoint = subscription.environment.resourceManagerEndpointUrl.replace(/\/$/, "");
    const url = new URL(`${endpoint}/subscriptions/${encodeURIComponent(subscriptionID)}/locations`);
    url.searchParams.set("api-version", "2022-12-01");
    const payload = await this.armValuePages(subscription, url);
    return parseLocations(payload);
  }

  public async recommendationData(subscriptionID: string, location: string): Promise<AzureRecommendationData> {
    if (!/^[a-z0-9-]{1,64}$/i.test(location)) {
      throw new Error("The Azure region name is invalid.");
    }
    const subscription = await this.subscription(subscriptionID);
    const endpoint = subscription.environment.resourceManagerEndpointUrl.replace(/\/$/, "");
    const base = `${endpoint}/subscriptions/${encodeURIComponent(subscriptionID)}/providers`;
    const postgresCapabilities = new URL(`${base}/Microsoft.DBforPostgreSQL/locations/${encodeURIComponent(location)}/capabilities`);
    postgresCapabilities.searchParams.set("api-version", "2024-08-01");
    const computeSkus = new URL(`${base}/Microsoft.Compute/skus`);
    computeSkus.searchParams.set("api-version", "2021-07-01");
    computeSkus.searchParams.set("$filter", `location eq '${location}'`);
    const postgresQuota = new URL(`${base}/Microsoft.DBforPostgreSQL/locations/${encodeURIComponent(location)}/resourceType/flexibleServers/usages`);
    postgresQuota.searchParams.set("api-version", "2025-08-01");
    const computeQuota = new URL(`${base}/Microsoft.Compute/locations/${encodeURIComponent(location)}/usages`);
    computeQuota.searchParams.set("api-version", "2025-04-01");
    const [postgresPayload, computePayload, postgresQuotaPayload, computeQuotaPayload] = await Promise.all([
      this.armValuePages(subscription, postgresCapabilities),
      this.armValuePages(subscription, computeSkus),
      this.armValuePages(subscription, postgresQuota),
      this.armValuePages(subscription, computeQuota)
    ]);
    return {
      postgres: parsePostgresCapabilities(postgresPayload),
      computeSkus: parseComputeSkus(computePayload, location),
      postgresQuota: parseQuotaUsages(postgresQuotaPayload),
      computeQuota: parseQuotaUsages(computeQuotaPayload)
    };
  }

  public async retailRates(location: string, skuNames: string[]): Promise<RetailRate[]> {
    if (!/^[a-z0-9-]{1,64}$/i.test(location) || skuNames.some((sku) => !/^[A-Za-z0-9_]{1,128}$/.test(sku))) {
      throw new Error("The retail-price lookup parameters are invalid.");
    }
    const filters = skuNames.map((sku) => `armSkuName eq '${sku}'`).join(" or ");
    const url = new URL("https://prices.azure.com/api/retail/prices");
    url.searchParams.set("currencyCode", "USD");
    url.searchParams.set("$filter", `armRegionName eq '${location}' and priceType eq 'Consumption' and (${filters})`);
    const items: unknown[] = [];
    let nextURL: URL | undefined = url;
    for (let pageNumber = 0; nextURL && pageNumber < 20; pageNumber += 1) {
      if (nextURL.protocol !== "https:" || nextURL.hostname !== "prices.azure.com") {
        throw new Error("Azure Retail Prices returned an unsafe continuation link.");
      }
      const response = await fetch(nextURL, { signal: AbortSignal.timeout(30_000) });
      if (!response.ok) {
        throw new Error(`Azure Retail Prices returned ${response.status}.`);
      }
      const payload = await response.json() as Record<string, unknown>;
      if (!Array.isArray(payload.Items)) {
        throw new Error("Azure Retail Prices returned an invalid response.");
      }
      items.push(...payload.Items);
      nextURL = typeof payload.NextPageLink === "string" && payload.NextPageLink
        ? new URL(payload.NextPageLink)
        : undefined;
    }
    if (nextURL) {
      throw new Error("Azure Retail Prices pagination exceeded the safety limit.");
    }
    return parseRetailRates({ Items: items });
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

  private async armResourcePages(subscription: AzureSubscription, firstURL: URL): Promise<ResourceSummary[]> {
    const payload = await this.armPages(subscription, firstURL, (value) => {
      const page = parseResourcePage(value);
      return { items: page.resources, nextLink: page.nextLink };
    });
    return payload;
  }

  private async armValuePages(subscription: AzureSubscription, firstURL: URL): Promise<{ value: unknown[] }> {
    const values = await this.armPages(subscription, firstURL, (value) => {
      if (value === null || typeof value !== "object" || !Array.isArray((value as { value?: unknown }).value)) {
        throw new Error("Azure Resource Manager returned an invalid paged response.");
      }
      const page = value as { value: unknown[]; nextLink?: unknown };
      if (page.nextLink !== undefined && typeof page.nextLink !== "string") {
        throw new Error("Azure Resource Manager returned an invalid continuation link.");
      }
      return { items: page.value, nextLink: page.nextLink };
    });
    return { value: values };
  }

  private async armPages<T>(
    subscription: AzureSubscription,
    firstURL: URL,
    parse: (value: unknown) => { items: T[]; nextLink?: string }
  ): Promise<T[]> {
    const endpoint = subscription.environment.resourceManagerEndpointUrl.replace(/\/$/, "");
    const endpointOrigin = new URL(endpoint).origin;
    const token = await subscription.credential.getToken(`${endpoint}/.default`);
    if (!token?.token) {
      throw new Error("VS Code could not obtain an Azure Resource Manager token.");
    }
    const items: T[] = [];
    let nextURL: URL | undefined = firstURL;
    for (let pageNumber = 0; nextURL && pageNumber < 100; pageNumber += 1) {
      if (nextURL.protocol !== "https:" || nextURL.origin !== endpointOrigin) {
        throw new Error("Azure Resource Manager returned an unsafe continuation link.");
      }
      const response = await fetch(nextURL, {
        headers: { authorization: `Bearer ${token.token}` },
        signal: AbortSignal.timeout(30_000)
      });
      if (!response.ok) {
        throw new Error(`Azure Resource Manager returned ${response.status}.`);
      }
      const page = parse(await response.json() as unknown);
      items.push(...page.items);
      nextURL = page.nextLink ? new URL(page.nextLink) : undefined;
    }
    if (nextURL) {
      throw new Error("Azure Resource Manager pagination exceeded the safety limit.");
    }
    return items;
  }
}
