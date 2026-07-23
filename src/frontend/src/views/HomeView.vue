<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { uploadFile, getTaskStatus, type PipelineStatus } from '@/api/knowledge'

const router = useRouter()
const selectedFile = ref<File | null>(null)
const isUploading = ref(false)
const uploadStatus = ref('')
const taskStatus = ref<PipelineStatus | null>(null)
const errorMessage = ref('')
const currentFileId = ref<string>('')

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    selectedFile.value = target.files[0]
    errorMessage.value = ''
  }
}

const handleUpload = async () => {
  if (!selectedFile.value) {
    errorMessage.value = '请选择文件'
    return
  }

  isUploading.value = true
  uploadStatus.value = '开始上传...'
  errorMessage.value = ''

  try {
    // 上传文件
    const uploadRes = await uploadFile(selectedFile.value)
    uploadStatus.value = '上传成功，开始处理...'
    currentFileId.value = uploadRes.file_id
    
    if (uploadRes.file_id) {
      // 轮询任务状态
      await pollTaskStatus(uploadRes.file_id)
    } else {
      uploadStatus.value = '处理完成！'
    }
  } catch (error: any) {
    console.error('Upload error:', error)
    errorMessage.value = error.response?.data?.detail || '上传失败，请重试'
    isUploading.value = false
  }
}

const pollTaskStatus = async (fileId: string) => {
  const maxAttempts = 60
  let attempts = 0

  while (attempts < maxAttempts) {
    try {
      const status = await getTaskStatus(fileId)
      taskStatus.value = status
      uploadStatus.value = `处理中... ${status.progress}% - ${status.message || ''}`

      if (status.status === 'completed' || status.progress >= 100) {
        uploadStatus.value = '处理完成！'
        isUploading.value = false
        break
      } else if (status.status === 'failed') {
        throw new Error(status.message || '处理失败')
      }

      attempts++
      await new Promise(resolve => setTimeout(resolve, 2000))
    } catch (error: any) {
      console.error('Polling error:', error)
      errorMessage.value = error.message || '查询状态失败'
      isUploading.value = false
      break
    }
  }

  if (attempts >= maxAttempts) {
    uploadStatus.value = '处理时间较长，请稍后查看'
    isUploading.value = false
  }
}

const goToSearch = () => {
  router.push('/search')
}
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
    <div class="container mx-auto px-4 py-16">
      <div class="max-w-2xl mx-auto">
        <h1 class="text-4xl font-bold text-center text-gray-800 mb-8">
          知识库增量更新
        </h1>
        
        <div class="bg-white rounded-xl shadow-lg p-8">
          <div class="mb-6">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              选择文件（支持 PDF、Word、PPT）
            </label>
            <div class="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-indigo-500 transition-colors">
              <div class="space-y-1 text-center">
                <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                  <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                <div class="flex text-sm text-gray-600 justify-center">
                  <label class="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none">
                    <span>上传文件</span>
                    <input 
                      type="file" 
                      class="sr-only" 
                      accept=".pdf,.doc,.docx,.ppt,.pptx"
                      @change="handleFileChange"
                    />
                  </label>
                  <p class="pl-1">或拖拽到此处</p>
                </div>
                <p class="text-xs text-gray-500">
                  PDF, DOC, DOCX, PPT, PPTX 最大 50MB
                </p>
              </div>
            </div>
            
            <div v-if="selectedFile" class="mt-4 p-3 bg-gray-50 rounded-lg">
              <p class="text-sm text-gray-700">
                <span class="font-medium">已选择:</span> {{ selectedFile.name }}
                <span class="text-gray-500 ml-2">({{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB)</span>
              </p>
            </div>
          </div>

          <button
            @click="handleUpload"
            :disabled="!selectedFile || isUploading"
            class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {{ isUploading ? '处理中...' : '上传并处理' }}
          </button>

          <div v-if="uploadStatus" class="mt-6">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p class="text-sm text-blue-800">{{ uploadStatus }}</p>
              <div v-if="isUploading" class="mt-3">
                <div class="w-full bg-blue-200 rounded-full h-2">
                  <div 
                    class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    :style="{ width: `${taskStatus?.progress || 0}%` }"
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="errorMessage" class="mt-6">
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
              <p class="text-sm text-red-800">{{ errorMessage }}</p>
            </div>
          </div>

          <div v-if="!isUploading && uploadStatus && !errorMessage" class="mt-6">
            <button
              @click="goToSearch"
              class="w-full flex justify-center py-3 px-4 border border-indigo-600 rounded-md shadow-sm text-sm font-medium text-indigo-600 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
            >
              前往检索页面
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
</style>
