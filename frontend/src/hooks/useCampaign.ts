import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function useDesigns() {
  return useQuery({
    queryKey: ["designs"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/designs");
      if (error) throw error;
      return data;
    },
  });
}

export function useCampaignSummary(design: string, platform: string) {
  return useQuery({
    queryKey: ["campaign-summary", design, platform],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/campaigns/{design}/{platform}", {
        params: { path: { design, platform } },
      });
      if (error) throw error;
      return data;
    },
    enabled: Boolean(design && platform),
  });
}

export function useCampaignPareto(design: string, platform: string) {
  return useQuery({
    queryKey: ["campaign-pareto", design, platform],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/campaigns/{design}/{platform}/pareto", {
        params: { path: { design, platform } },
      });
      if (error) throw error;
      return data;
    },
    enabled: Boolean(design && platform),
  });
}

export function useCampaignFunnel(design: string, platform: string) {
  return useQuery({
    queryKey: ["campaign-funnel", design, platform],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/campaigns/{design}/{platform}/funnel", {
        params: { path: { design, platform } },
      });
      if (error) throw error;
      return data;
    },
    enabled: Boolean(design && platform),
  });
}

export function useCampaignHistory(design: string, platform: string) {
  return useQuery({
    queryKey: ["campaign-history", design, platform],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/campaigns/{design}/{platform}/history", {
        params: { path: { design, platform } },
      });
      if (error) throw error;
      return data;
    },
    enabled: Boolean(design && platform),
  });
}

export function useCampaignBest(design: string, platform: string, top = 3) {
  return useQuery({
    queryKey: ["campaign-best", design, platform, top],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/campaigns/{design}/{platform}/best", {
        params: { path: { design, platform }, query: { top } },
      });
      if (error) throw error;
      return data;
    },
    enabled: Boolean(design && platform),
  });
}
