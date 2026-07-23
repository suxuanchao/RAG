import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export interface UploadResponse {
  status: string
  message: string
  file_name: string
  file_id: string
  stages_completed: string[]
}

export interface SearchRequest {
  query: string
  top_k?: number
  filters?: Record<string, any>
}

export interface SearchResult {
  content: string
  doc_name?: string
  headers?: string
  score: number
  trust_score: number
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
}

export interface PipelineStatus {
  file_id: string
  stage: string
  status: string
  message: string
  progress: number
}

// 上传文件
export const uploadFile = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post<UploadResponse>('/upload_document', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

// 查询任务状态
export const getTaskStatus = async (fileId: string): Promise<PipelineStatus> => {
  const response = await api.get<PipelineStatus>(`/pipeline_status/${fileId}`)
  return response.data
}

// 知识库检索
export const searchKnowledge = async (
  query: string,
  topK: number = 5,
  filters?: Record<string, any>
): Promise<SearchResponse> => {
  const params: Record<string, any> = {
    question: query,
    top_k: topK,
  }
  if (filters) {
    params.filter_json = JSON.stringify(filters)
  }
  const response = await api.get<SearchResponse>('/query', { params })
  return response.data
}

// 健康检查
export const healthCheck = async (): Promise<any> => {
  const response = await api.get('/health')
  return response.data
}

// 获取统计信息
export const getStats = async (): Promise<any> => {
  const response = await api.get('/stats')
  return response.data
}
